# recommendation.py

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from itertools import combinations
import numpy as np

# Load and preprocess data
df = pd.read_csv('data_wisata_fix.csv')
df = df.drop(columns=['Rentang Harga', 'Lokasi / Tempat'], axis=1)

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude'])
df['Harga'] = pd.to_numeric(df['Harga'], errors='coerce')
df['Koordinat'] = list(zip(df['latitude'], df['longitude']))

# Filter data function
def filter_data(df, user_kategori, user_jenis_wisata, user_child_friendly):
    filter_kategori = df[df['Kategori'].str.contains(user_kategori, case=False, na=False)]
    filter_jenis = filter_kategori[filter_kategori['Jenis Wisata'].str.contains('|'.join(user_jenis_wisata), case=False, na=False)]
    if user_child_friendly.lower() == 'yes':
        filter_child = filter_jenis[filter_jenis['Child Friendly'].str.contains('Yes', case=False, na=False)]
    else:
        filter_child = filter_jenis

    if filter_child.empty:
        filter_child = df[df['Jenis Wisata'].str.contains('|'.join(user_jenis_wisata), case=False, na=False)]

    return filter_child

class TripModel(tfrs.Model):
    def __init__(self, name='trip_model', **kwargs):
        super().__init__(name=name, **kwargs)

        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=df["Jenis Wisata"].unique(), mask_token=None),
            tf.keras.layers.Embedding(len(df["Jenis Wisata"].unique()) + 1, 32)
        ])

        self.description_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=df["Deskripsi"].unique(), mask_token=None),
            tf.keras.layers.Embedding(len(df["Deskripsi"].unique()) + 1, 32)
        ])

        self.facilities_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=df["Fasilitas"].unique(), mask_token=None),
            tf.keras.layers.Embedding(len(df["Fasilitas"].unique()) + 1, 32)
        ])

        self.accessibility_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=df["Aksesibilitas"].unique(), mask_token=None),
            tf.keras.layers.Embedding(len(df["Aksesibilitas"].unique()) + 1, 32)
        ])

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features, training=False):
        item_embeddings = self.item_embedding(features["Jenis Wisata"])
        description_embeddings = self.description_embedding(features["Deskripsi"])
        facilities_embeddings = self.facilities_embedding(features["Fasilitas"])
        accessibility_embeddings = self.accessibility_embedding(features["Aksesibilitas"])

        embeddings = tf.concat([item_embeddings, description_embeddings, facilities_embeddings, accessibility_embeddings], axis=1)
        ratings = self.rating_model(embeddings)

        return self.task(
            labels=features["Rating"],
            predictions=ratings
        )

features = {
    "Jenis Wisata": df["Jenis Wisata"].values,
    "Deskripsi": df["Deskripsi"].values,
    "Fasilitas": df["Fasilitas"].values,
    "Aksesibilitas": df["Aksesibilitas"].values,
    "Rating": df["Rating"].values
}

model = TripModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(features, epochs=10)


def filter_and_recommend(user_kategori, user_jenis_wisata, user_child_friendly, user_budget, user_latitude, user_longitude, num_recommendations):
    filtered_data = filter_data(df, user_kategori, user_jenis_wisata, user_child_friendly)

    def calculate_distance(coord1, coord2):
        return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

    user_coords = (user_latitude, user_longitude)
    filtered_data["Distance"] = filtered_data["Koordinat"].apply(lambda x: calculate_distance(user_coords, x))

    features_for_prediction = {
        "Jenis Wisata": filtered_data["Jenis Wisata"].values,
        "Deskripsi": filtered_data["Deskripsi"].values,
        "Fasilitas": filtered_data["Fasilitas"].values,
        "Aksesibilitas": filtered_data["Aksesibilitas"].values
    }

    item_embeddings = model.item_embedding(features_for_prediction["Jenis Wisata"])
    description_embeddings = model.description_embedding(features_for_prediction["Deskripsi"])
    facilities_embeddings = model.facilities_embedding(features_for_prediction["Fasilitas"])
    accessibility_embeddings = model.accessibility_embedding(features_for_prediction["Aksesibilitas"])

    embeddings = tf.concat([item_embeddings, description_embeddings, facilities_embeddings, accessibility_embeddings], axis=1)

    if filtered_data.empty:
        predicted_ratings = model.rating_model(embeddings).numpy().flatten()
        filter_kategori = df[df['Kategori'].str.contains(user_kategori, case=False, na=False)]
        filter_kategori["Predicted Rating"] = predicted_ratings
        recommendations = filter_kategori[filter_kategori['Harga'] <= user_budget]
    else:
        predicted_ratings = model.rating_model(embeddings).numpy().flatten()
        filtered_data["Predicted Rating"] = predicted_ratings
        recommendations = filtered_data[filtered_data['Harga'] <= user_budget]

    if recommendations.empty:
        print("Tidak ada rekomendasi dalam budget yang diberikan.")
        recommendations = filtered_data.sort_values(by='Harga').head(num_recommendations)
        return recommendations

    recommendations["Distance"] = recommendations["Koordinat"].apply(lambda x: calculate_distance(user_coords, x))
    recommendations = recommendations.sort_values(by="Distance").reset_index(drop=True)

    best_combination = None
    best_rating_sum = -np.inf
    best_distance_sum = np.inf
    combination_count = 0
    max_combinations = 1000

    for r in range(1, num_recommendations + 1):
        if combination_count > max_combinations:
            break
        for combination in combinations(recommendations.index, r):
            combination_count += 1
            if combination_count > max_combinations:
                break
            total_cost = recommendations.loc[list(combination), 'Harga'].sum()
            if total_cost <= user_budget:
                rating_sum = recommendations.loc[list(combination), 'Predicted Rating'].sum()
                distance_sum = recommendations.loc[list(combination), 'Distance'].sum()
                if rating_sum > best_rating_sum or (rating_sum == best_rating_sum and distance_sum < best_distance_sum):
                    best_rating_sum = rating_sum
                    best_distance_sum = distance_sum
                    best_combination = combination

    if best_combination:
        return recommendations.loc[list(best_combination)].sort_values(by='Predicted Rating', ascending=False)
    else:
        print("Tidak ada tempat wisata yang sesuai dengan preferensi yang diberikan.")
        return pd.DataFrame()
