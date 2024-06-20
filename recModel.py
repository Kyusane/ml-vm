import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate('./maliva-ml-service.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load data from Firestore
def load_data_from_firestore():
    # Mengambil data tempat wisata dari Firestore
    docs = db.collection('Destinations').stream()
    data = []

    for doc in docs:
        tempat_wisata = doc.to_dict()
        tempat_wisata['id'] = doc.id 
        data.append(tempat_wisata)

    return pd.DataFrame(data)

# Load data
data = load_data_from_firestore()

# Preprocessing
def preprocess_text(text):
    return text.replace(", ", "").replace(" ", "")

data['Aksesibilitas'] = data['Aksesibilitas'].apply(preprocess_text)
data['Fasilitas'] = data['Fasilitas'].apply(preprocess_text)

# Tokenisasi deskripsi
desc_tokenizer = Tokenizer(num_words=10000)
desc_tokenizer.fit_on_texts(data['Deskripsi'])
desc_sequences = desc_tokenizer.texts_to_sequences(data['Deskripsi'])
desc_data = pad_sequences(desc_sequences, maxlen=100)

# Tokenisasi aksesibilitas
acc_tokenizer = Tokenizer(num_words=1000)
acc_tokenizer.fit_on_texts(data['Aksesibilitas'])
acc_sequences = acc_tokenizer.texts_to_sequences(data['Aksesibilitas'])
acc_data = pad_sequences(acc_sequences, maxlen=10)

# Tokenisasi jenis wisata
type_tokenizer = Tokenizer(num_words=100)
type_tokenizer.fit_on_texts(data['Jenis Wisata'])
type_sequences = type_tokenizer.texts_to_sequences(data['Jenis Wisata'])
type_data = pad_sequences(type_sequences, maxlen=5)

# Tokenisasi fasilitas yang tersedia
fac_tokenizer = Tokenizer(num_words=1000)
fac_tokenizer.fit_on_texts(data['Fasilitas'])
fac_sequences = fac_tokenizer.texts_to_sequences(data['Fasilitas'])
fac_data = pad_sequences(fac_sequences, maxlen=20)

# Definisikan input untuk deskripsi
desc_input = Input(shape=(100,), name='desc_input')
desc_embedding = Embedding(input_dim=10000, output_dim=64, input_length=100, name='desc_embedding')(desc_input)
desc_flatten = Flatten()(desc_embedding)

# Definisikan input untuk aksesibilitas
acc_input = Input(shape=(10,), name='acc_input')
acc_embedding = Embedding(input_dim=1000, output_dim=32, input_length=10, name='acc_embedding')(acc_input)
acc_flatten = Flatten()(acc_embedding)

# Definisikan input untuk jenis wisata
type_input = Input(shape=(5,), name='type_input')
type_embedding = Embedding(input_dim=100, output_dim=16, input_length=5, name='type_embedding')(type_input)
type_flatten = Flatten()(type_embedding)

# Definisikan input untuk fasilitas yang tersedia
fac_input = Input(shape=(20,), name='fac_input')
fac_embedding = Embedding(input_dim=1000, output_dim=32, input_length=20, name='fac_embedding')(fac_input)
fac_flatten = Flatten()(fac_embedding)

# Gabungkan semua embedding
merged = Concatenate()([desc_flatten, acc_flatten, type_flatten, fac_flatten])

# Membangun model embedding
embedding_model = Model(inputs=[desc_input, acc_input, type_input, fac_input], outputs=merged)

# Mendapatkan vektor embedding untuk tempat wisata
def get_place_embeddings():
    place_embeddings = []
    docs = db.collection('Destinations').stream()

    for doc in docs:
        tempat_wisata = doc.to_dict()
        desc_seq = desc_tokenizer.texts_to_sequences([tempat_wisata['Deskripsi']])
        acc_seq = acc_tokenizer.texts_to_sequences([tempat_wisata['Aksesibilitas']])
        type_seq = type_tokenizer.texts_to_sequences([tempat_wisata['Jenis Wisata']])
        fac_seq = fac_tokenizer.texts_to_sequences([tempat_wisata['Fasilitas']])

        desc_data = pad_sequences(desc_seq, maxlen=100)
        acc_data = pad_sequences(acc_seq, maxlen=10)
        type_data = pad_sequences(type_seq, maxlen=5)
        fac_data = pad_sequences(fac_seq, maxlen=20)

        place_embedding = embedding_model.predict([desc_data, acc_data, type_data, fac_data])
        place_embeddings.append(place_embedding)

    return place_embeddings

place_embeddings = get_place_embeddings()

# Fungsi untuk mendapatkan riwayat pencarian dari semua pengguna
def get_all_users_search_history_from_firestore():
    # Query ke Firestore untuk mendapatkan riwayat pencarian dari semua pengguna
    docs = db.collection('SearchHistory').stream()
    search_terms = []
    
    for doc in docs:
        search_term = doc.to_dict()['search']
        search_terms.append(search_term)

    return search_terms

def get_all_users_preference_vector(input_embedding):
    search_terms = get_all_users_search_history_from_firestore()
    if search_terms:
        search_desc_sequences = desc_tokenizer.texts_to_sequences(search_terms)
        search_desc_data = pad_sequences(search_desc_sequences, maxlen=100)
        search_acc_data = np.zeros((len(search_terms), 10))  # Jika tidak ada data aksesibilitas
        search_type_data = np.zeros((len(search_terms), 5))  # Jika tidak ada data jenis wisata
        search_fac_sequences = fac_tokenizer.texts_to_sequences(search_terms)
        search_fac_data = pad_sequences(search_fac_sequences, maxlen=20)

        search_embeddings = embedding_model.predict([search_desc_data, search_acc_data, search_type_data, search_fac_data])
        averaged_embedding = np.mean(search_embeddings, axis=0, keepdims=True)

        # Padding atau normalisasi dimensi agar cocok dengan input_embedding
        # Misalnya, melakukan padding dengan nilai nol untuk mencocokkan dimensi
        padded_preference_vector = np.zeros_like(input_embedding)  # Dimensi input_embedding
        padded_preference_vector[:, :averaged_embedding.shape[1]] = averaged_embedding

        return padded_preference_vector
    else:
        return np.zeros_like(input_embedding)  # Sesuaikan dengan dimensi input_embedding Anda

# Fungsi utama untuk rekomendasi
def recommend_with_all_users_history(input_name, top_n=10):
    place_index = data[data['Nama Wisata'].str.contains(input_name, case=False)].index
    if len(place_index) > 0:
        place_index = place_index[0]
        input_seq_desc = desc_tokenizer.texts_to_sequences([data.loc[place_index, 'Deskripsi']])
        input_seq_acc = acc_tokenizer.texts_to_sequences([data.loc[place_index, 'Aksesibilitas']])
        input_seq_type = type_tokenizer.texts_to_sequences([data.loc[place_index, 'Jenis Wisata']])
        input_seq_fac = fac_tokenizer.texts_to_sequences([data.loc[place_index, 'Fasilitas']])

        input_data_desc = pad_sequences(input_seq_desc, maxlen=100)
        input_data_acc = pad_sequences(input_seq_acc, maxlen=10)
        input_data_type = pad_sequences(input_seq_type, maxlen=5)
        input_data_fac = pad_sequences(input_seq_fac, maxlen=20)

        input_embedding = embedding_model.predict([input_data_desc, input_data_acc, input_data_type, input_data_fac])
    else:
        input_seq_type = type_tokenizer.texts_to_sequences([input_name])
        input_data_type = pad_sequences(input_seq_type, maxlen=5)
        input_seq_fac = fac_tokenizer.texts_to_sequences([input_name])
        input_data_fac = pad_sequences(input_seq_fac, maxlen=20)

        input_embedding = embedding_model.predict([np.zeros((1, 100)), np.zeros((1, 10)), input_data_type, input_data_fac])

    all_users_preference_vector = get_all_users_preference_vector(input_embedding)

    # Ubah cara Anda menggabungkan input_embedding dan all_users_preference_vector
    combined_embedding = 0.5 * np.squeeze(input_embedding, axis=0) + 0.5 * all_users_preference_vector
    combined_embedding = combined_embedding.reshape(1, -1)  # Pastikan ini 2D

    similarities = cosine_similarity(combined_embedding, np.concatenate(place_embeddings, axis=0))
    similar_indices = similarities.argsort()[0][-top_n-1:][::-1]
    recommendations = data.iloc[similar_indices].to_dict(orient='records')
    return recommendations