from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# تحميل النموذج
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # جمع البيانات من النموذج
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    # إنشاء DataFrame للإدخال
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # التوقع
    prediction = model.predict(input_data)[0]

    # تنسيق النتيجة (دون كسور عشرية)
    prediction_formatted = f"{prediction:,.2f}"

    return render_template('index.html', prediction=prediction_formatted)

if __name__ == '__main__':
    app.run(debug=True)
