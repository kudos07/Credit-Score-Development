from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('notebooks/model.pkl')  # Update with your model path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        sex = float(request.form['sex'])
        marriage = float(request.form['marriage'])
        age = float(request.form['age'])
        bill_amt1 = float(request.form['bill_amt1'])
        education = float(request.form['education'])
        pay_1 = float(request.form['pay_1'])

        # Debug information
        print(f"Inputs: {sex}, {marriage}, {age}, {bill_amt1}, {education}, {pay_1}")

        # Make prediction using the loaded model
        input_values = np.array([[sex, marriage, age, bill_amt1, education, pay_1]])
        prediction = model.predict(input_values)[0]

        # Debug information
        print(f"Prediction: {prediction}")

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
