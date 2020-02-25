# app/app.py

# Common python package imports.
from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import xgboost as xgb
# Import from app/features.py.
from features import FEATURES

# Initialize the app and set a secret_key.
app = Flask(__name__)
app.secret_key = 'something_secret'

# Load the pickled model.

MODEL = pickle.load(open('model.pkl', 'rb'))


LABELS = {'0':'setosa', '1':'versicolor', '2':'virginica'}

# app/app.py (continued)

@app.route('/api', methods=['GET'])
def api():
    """Handle request and output model score in json format."""

    # Handle empty requests.
    if not request.json:
        return jsonify({'error': 'no request received'})

    # Parse request args into feature array for prediction.
    x_list, missing_data = parse_args(request.json)
    x_array = np.array([x_list])
    x_array = x_array.reshape(1, -1)

    # Predict on x_array and return JSON response.
    darray = xgb.DMatrix(x_array)
    predictions = MODEL.predict(darray)
    estimate = np.asarray([np.argmax(line) for line in predictions])[0]
    response = dict(PREDICTION=LABELS[str(estimate)])
    return jsonify(response)

def parse_args(request_dict):
    """Parse model features from incoming requests formatted in
    JSON."""
    # Initialize missing_data as False.
    missing_data = False# Parse out the features from the request_dict.
    x_list = []
    for feature in FEATURES:
        value = request_dict.get(feature, None)
        if value:
            x_list.append(value)
        else:
            # Handle missing features.
            x_list.append(0)
            missing_data = True
    return x_list, missing_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
