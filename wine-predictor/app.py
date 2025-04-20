 from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('model.joblib')
features = joblib.load('features.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Prepare input with defaults
        input_data = {f: float(data.get(f, 0)) for f in features}
        
        # Make prediction
        prediction = model.predict(pd.DataFrame([input_data]))[0]
        proba = model.predict_proba(pd.DataFrame([input_data]))[0][1 if prediction == "Good" else 0]
        
        return jsonify({
            "prediction": prediction,
            "confidence": f"{proba:.0%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)