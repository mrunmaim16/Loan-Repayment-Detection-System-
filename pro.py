from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# Define a route to serve the HTML page
    
def preprocess_data(data):
    # Convert categorical variables to one-hot encoding
    code_gender = 1 if data['CODE_GENDER'] == 'M' else 0
    name_family_status_mapping = {
    'Civil marriage': 0,
    'Married': 1,
    'Separated': 2,
    'Single': 3,
    'Unknown': 4,
    'Widow': 5
}
    name_family_status = name_family_status_mapping[data['NAME_FAMILY_STATUS']]
    #name_family_status = 1 if data['NAME_FAMILY_STATUS'] == 'single' else 0
    
    # Convert other variables to float and normalize if necessary
    cnt_children = float(data['CNT_CHILDREN'])
    amt_income_total = float(data['AMT_INCOME_TOTAL'])
    amt_credit = float(data['AMT_CREDIT'])
    amt_annuity = float(data['AMT_ANNUITY'])
    days_birth = float(data['DAYS_BIRTH'])
    days_employed = float(data['DAYS_EMPLOYED'])
    
    # Return a numpy array of the preprocessed data
    return np.array([
        code_gender,
        cnt_children,
        amt_income_total,
        amt_credit,
        amt_annuity,
        days_birth,
        days_employed,
        name_family_status
    ]).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            'CODE_GENDER': request.form['CODE_GENDER'],
            'CNT_CHILDREN': request.form['CNT_CHILDREN'],
            'AMT_INCOME_TOTAL': request.form['AMT_INCOME_TOTAL'],
            'AMT_CREDIT': request.form['AMT_CREDIT'],
            'AMT_ANNUITY': request.form['AMT_ANNUITY'],
            'DAYS_BIRTH': request.form['DAYS_BIRTH'],
            'DAYS_EMPLOYED': request.form['DAYS_EMPLOYED'],
            'NAME_FAMILY_STATUS': request.form['NAME_FAMILY_STATUS']
        }
        features = preprocess_data(data)
        prediction = model.predict(features)[0]
        print("Prediction:", prediction)
        if prediction==1:
            return "Prediction: " + str(prediction) + " Person won't be able to repay the loan."
        else:
            return "Prediction: " + str(prediction) +" Person will repay the loan."
if __name__ == '__main__':
    app.run(debug=False,port=5001)

