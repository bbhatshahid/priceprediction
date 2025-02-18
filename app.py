from flask import Flask , request, jsonify, render_template
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)
model = joblib.load('rfmodel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the HTML form
    data = request.form
    features = [
        float(data['bedrooms']),
        float(data['bathrooms']),
        float(data['livingarea']),
        float(data['floors']),
        float(data['arhouse']),
        float(data['builtyr']),
        float(data['lotarea']),
        float(data['grade']),
        float(data['waterfront']),
    ]
    # Convert to a numpy array and reshape for prediction
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)

     # Return the prediction as JSON or render back to the HTML page
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)