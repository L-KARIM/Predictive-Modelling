from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('heart_failure_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form.get('age'))
        anaemia = int(request.form.get('anaemia'))
        creatinine_phosphokinase = float(request.form.get('creatinine_phosphokinase'))
        diabetes = int(request.form.get('diabetes'))
        ejection_fraction = float(request.form.get('ejection_fraction'))
        high_blood_pressure = int(request.form.get('high_blood_pressure'))
        platelets = float(request.form.get('platelets'))
        serum_creatinine = float(request.form.get('serum_creatinine'))
        serum_sodium = float(request.form.get('serum_sodium'))
        sex = int(request.form.get('sex'))
        smoking = int(request.form.get('smoking'))
        time = float(request.form.get('time'))
        
        # Create feature array
        features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, 
                            ejection_fraction, high_blood_pressure, platelets, 
                            serum_creatinine, serum_sodium, sex, smoking, time]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Prepare result
        if prediction == 1:
            result = "High Risk"
            risk_percentage = prediction_proba[1] * 100
            message = f"The model predicts a high risk of heart failure with {risk_percentage:.1f}% confidence."
            alert_class = "alert-danger"
        else:
            result = "Low Risk"
            risk_percentage = prediction_proba[0] * 100
            message = f"The model predicts a low risk of heart failure with {risk_percentage:.1f}% confidence."
            alert_class = "alert-success"
        
        return render_template('index.html', 
                             result=result, 
                             message=message, 
                             alert_class=alert_class,
                             risk_percentage=risk_percentage)
    
    except Exception as e:
        error_message = f"Error in prediction: {str(e)}"
        return render_template('index.html', 
                             error=error_message, 
                             alert_class="alert-warning")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

