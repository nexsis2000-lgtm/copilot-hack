# Create a base Flask server

import pickle
import csv
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

# Enable cors
@app.after_request
def after_request(response):
    """
    Enable CORS
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


# Load model from pickle file
ASSET_DIR = Path(__file__).resolve().parent
with (ASSET_DIR / 'model.pkl').open('rb') as model_file:
    model = pickle.load(model_file)

# Model takes two parameters - day of week and airport id, then returns a prediction of flight delay
@app.route('/predict', methods=['GET'])
def predict():
    """
    Takes two parameters - day of week and airport id, then returns a prediction of flight delay
    """
    # Store day_of_week as int
    try:
        day_of_week = int(request.args.get('day_of_week', ''))
        airport_id = int(request.args.get('airport_id', ''))
    except (TypeError, ValueError):
        return jsonify({'error': 'day_of_week and airport_id must be integers'}), 400
    if not 1 <= day_of_week <= 7 or airport_id <= 0:
        return jsonify({'error': 'day_of_week must be 1–7 and airport_id must be positive'}), 400
    prediction = model.predict_proba([[day_of_week, airport_id]])[0]
    
    certainty, delay = (float(value) for value in prediction)

    # return prediction as json
    return jsonify({'certainty': certainty, 'delay': delay})

# Create a new route called airports with method of get
@app.route('/airports', methods=['GET'])
def airports():
    with (ASSET_DIR / 'airports.csv').open(newline='', encoding='utf-8') as source:
        rows = csv.reader(source)
        next(rows, None)
        airports = [{'id': int(row[0]), 'name': row[1]} for row in rows if row]
    # Sort by name
    airports = sorted(airports, key=lambda k: k['name'])

    return jsonify(airports)

if __name__ == '__main__':
    app.run(debug=True)
