from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the model and scaler
try:
    model = pickle.load(open('model/random_forest_model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    # In production, you might want to exit or handle this differently
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Welcome to Crop Prediction API",
        "endpoints": {
            "/predict": "POST - Make crop predictions",
            "/health": "GET - Check API health"
        },
        "example_payload": {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.87,
            "humidity": 82.00,
            "ph": 6.5,
            "rainfall": 202.93
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        # Convert to array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Get crop name
        crop_dict = {
            0: "apple", 1: "banana", 2: "blackgram", 3: "chickpea", 4: "coconut",
            5: "coffee", 6: "cotton", 7: "grapes", 8: "jute", 9: "kidneybeans",
            10: "lentil", 11: "maize", 12: "mango", 13: "mothbeans", 14: "mungbean",
            15: "muskmelon", 16: "orange", 17: "papaya", 18: "pigeonpeas", 19: "pomegranate",
            20: "rice", 21: "watermelon"
        }
        
        result = {
            "prediction": crop_dict[prediction[0]],
            "success": True
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Please ensure all required fields are provided with numeric values"
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)