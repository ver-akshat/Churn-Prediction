import pickle
import numpy as np

# Making a predict function
def predict(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

# Opening the saved model
with open('churn-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


customer = {
    'customerid': '5681-akshf',
    'gender': 'female',
    'seniorcitizen': 1,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 30,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'no',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 400,
    'totalcharges': 3620.75,
}
    

prediction = predict(customer, dv, model)

print('prediction: %.3f' % prediction)

if prediction >= 0.5:
    print('result: Churn')
else:
    print('result: Not churn')