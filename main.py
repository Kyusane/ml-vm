from flask import Flask, request, jsonify
from model import filter_and_recommend
from recModel import recommend_with_all_users_history

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({'status': 200,'message': 'Welcome to Maliva ML API'})

@app.route('/planner')
def planner():
    category = request.args.get('category')
    type = request.args.get('type')
    child = request.args.get('child')
    budget = request.args.get('budget', type=float)
    lat = request.args.get('lat', type=float)
    long = request.args.get('long', type=float)
    nrec = request.args.get('nrec', type=int)
    
    recommendations = filter_and_recommend(category, type, child, budget, lat, long, nrec)
    recommendations_list = recommendations.to_dict(orient='records')
    return jsonify({"code" : 200, "message" : "success" , "plan" : recommendations_list}),200

@app.route('/recommendations')
def recommendations():
    search = request.args.get('search')
    recommendations = recommend_with_all_users_history(search)
    return jsonify({"code" : 200, "message" : "success" , "recommendations" : recommendations}),200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
