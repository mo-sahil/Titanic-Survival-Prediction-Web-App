from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the feature names
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the AJAX request
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])

    # Prepare the input data as a DataFrame
    input_features = [pclass, sex, age, sibsp, parch, fare]
    features_df = pd.DataFrame([input_features], columns=feature_names)

    # Make the prediction
    prediction = model.predict(features_df)

    # Return the prediction result as JSON
    if prediction[0] == 1:
        prediction_text = 'Survived'
    else:
        prediction_text = 'Did not survive'

    return jsonify({'prediction': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)
