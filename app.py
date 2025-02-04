import joblib
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from Preprocessing_P import FillMissingValues
from Preprocessing_P import CreateMissingIndicators
from Preprocessing_P import VectorizeDescriptiveText
from Preprocessing_P import CollapseHighCardinality
from Preprocessing_P import OneHotEncodeLowCardinality
from Preprocessing_P import ConvertToIntegers
from Preprocessing_P import FeatureSelection
from Preprocessing_P import RandomForest
from Preprocessing_P import XGBoostClassification
from flask import Flask, Response, url_for

#storing the path for the pkl files 
randomforest_path = './model/random_forest_model_2.pkl'
xgboost_path= './model/xgboost_model.pkl'
preprocessing_path = './model/preprocessing_pipeline.pkl'

#loading the data from pkl files
with open(randomforest_path, 'rb') as rf_model_file:
    rf_model = joblib.load(rf_model_file)

with open(xgboost_path, 'rb') as xgb_model_file:
    xgb_model = joblib.load(xgb_model_file)

with open(preprocessing_path, 'rb') as preprocessing_file:
    preprocessor = joblib.load(preprocessing_file)

app = Flask(__name__)

def none_if_empty(arg):
    return None if arg == '' else arg

#home page which displays the form 
@app.route('/')
def home():
    return render_template('code.html') 

#GET request for predict url which predicts the job submitted
@app.route('/predict', methods=['GET'])
def predict():
    #getting all the user input values from the form
    title = request.args.get('job_title')
    description = none_if_empty(request.args.get('job-description'))
    salary_range = none_if_empty(request.args.get('salary-range'))
    location = none_if_empty(request.args.get('job-location'))
    requirements = none_if_empty(request.args.get('job-requirements'))
    department = none_if_empty(request.args.get('department'))
    company_profile = none_if_empty(request.args.get('company-profile'))
    benefits = none_if_empty(request.args.get('benefits'))
    required_education = none_if_empty(request.args.get('required-education'))
    required_experience = none_if_empty(request.args.get('required-experience'))
    industry = none_if_empty(request.args.get('industry'))
    function = none_if_empty(request.args.get('domain'))
    employment_type = none_if_empty(request.args.get('employment-type'))
    telecommuting = none_if_empty(request.args.get('telecommuting'))
    has_company_logo = none_if_empty(request.args.get('logo'))
    has_questions = none_if_empty(request.args.get('screening-questions'))

    inputs = [
        title, description, salary_range, location, requirements, department,
        company_profile, benefits, required_education, required_experience, industry, function,
        employment_type, telecommuting, has_company_logo, has_questions
    ]

    inputs = [x if x else np.nan for x in inputs]

    column_names = [
       'title', 'description', 'salary_range', 'location', 'requirements', 'department',
        'company_profile', 'benefits', 'required_education', 'required_experience', 'industry', 'function',
        'employment_type', 'telecommuting', 'has_company_logo', 'has_questions'
    ]
    
    df = pd.DataFrame([inputs], columns=column_names)
    print(df)
    print(df.columns)

    processed_features = preprocessor.transform(df)
    
    #retriving what model did the user choose (if the user didn't choose any model, the default model choosed is random forest)
    model_type = request.args.get('model_type', 'randomforest')  # Default is 'randomforest'
    print(f"model: {model_type}")

    if model_type == 'randomforest':
        model = rf_model
    elif model_type == 'xgboost':
        model = xgb_model
    else:
        return jsonify({'The model type selected is not valid'}), 400
    
    #passing the features i.e., the user inputs to the model for prediction
    prediction = model.predict(processed_features)

    image_url1 = url_for('static', filename='image2.jpg')#legitimate image
    image_url2 = url_for('static', filename='image3.jpg')#fraudulent image

    #if the prediction is '0' i.e., the job is legitimate
    if int(prediction[0]) == 0:
        predict = 'The job is Legitimate'
        html = f"""
        <html>
        <head><title>Prediction Result</title>
        <style>
        body{{
        display: flex;
                justify-content: center;
                align-items: center;
                 height: 100vh;
                background-image: url('{image_url2}');
        
        background-position: center; 
        text-align:center;
        }}
        .content {{
                width: 45%;
                border: 1px solid #ccc;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                background-color: rgba(255, 255, 255, 0.8); 
                padding: 40px;
            }}
        </style>
        </head>

        <body>
        <div class="content">
        <h1>Prediction Result</h1>
        <p><strong>Prediction:</strong> {predict}</p>
        <p><strong>Model Name:</strong> {model_type}</p>
        <p><strong>Job Title:</strong> {title}</p>
        <p><strong>Department:</strong> {department}</p>
        <p><strong>Location:</strong> {location}</p>
        </div>
        </body>
        </html>
        """
        #if the prediction is '1' i.e., the job is fraudulent
    else:
        predict = 'The job is Fraduluent'
        html= f"""
        <html>
        <head><title>Prediction Result</title>
        <style>
        body{{
        display: flex;
                justify-content: center;
                align-items: center;
                 height: 100vh;
        background-image: 
                    url('{image_url1}');
                  background-position: center; 
        text-align:center;    
        }}
        .content {{
                width: 70%; 
                border: 1px solid #ccc;
                padding: 20px;
                border-radius: 8px;
                
                background-color: white ; 
                padding: 40px;
            }}
        </style>
        </head>

        <body>
        <div class="content">
        <h1>Prediction Result</h1>
        <p><strong>Prediction:</strong> {predict}</p>
        <p><strong>Model Name:</strong> {model_type}</p>
        <p><strong>Job Title:</strong> {title}</p>
        <p><strong>Department:</strong> {department}</p>
        <p><strong>Location:</strong> {location}</p>
        </div>
        </body>
        </html>
        """
    return Response(html, mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
