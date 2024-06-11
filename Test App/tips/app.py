from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('tips/tip_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from POST request
    total_bill = float(request.form['total_bill'])
    size = int(request.form['size'])
    
    # Prepare the data for prediction
    data = {'total_bill': [total_bill], 'size': [size]}
    df = pd.DataFrame(data)
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    # Return the prediction result in rounded integer
    return render_template('results.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
