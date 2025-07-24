from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your model and scaler
model = joblib.load('pcos_model.pkl')
scaler = joblib.load('scaler.pkl')

# Route: Show the HTML form
@app.route('/')
def index():
    return render_template('form.html')

# Route: Handle form submission
@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        # Get data from the HTML form
        # Calculate BMI from height and weight
        weight = float(request.form['Weight (Kg)'])
        height_cm = float(request.form['Height(Cm)'])
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)

        input_features = [
            float(request.form['Age (yrs)']),
            weight,
            height_cm,
            bmi,
            int(request.form['Cycle(R/I)']),
            int(request.form['Fast food (Y/N)']),
            int(request.form['Pimples(Y/N)']),
            int(request.form['hair growth(Y/N)']),
            int(request.form['Skin darkening (Y/N)']),
            int(request.form['Follicle No. (L)']),
            int(request.form['Follicle No. (R)']),
            float(request.form['Avg. F size (L) (mm)']),
            float(request.form['Avg. F size (R) (mm)'])
        ]

        # Scale the input and make prediction
        input_array = np.array([input_features])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        result = "⚠️ PCOS Detected" if prediction == 1 else "✅ No PCOS Detected"

        return render_template('form.html', prediction=result)

    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")

# Run app
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use port 5001 if 5000 is taken

