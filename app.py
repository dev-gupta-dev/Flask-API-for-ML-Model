
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "Welcome to Iris Prediction API 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]

        # Map target to class name
        classes = ['Setosa', 'Versicolor', 'Virginica']
        result = classes[prediction]

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
