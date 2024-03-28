import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn import metrics
from sklearn.metrics import (auc,roc_auc_score, ConfusionMatrixDisplay, 
                             precision_score, PrecisionRecallDisplay,
                             recall_score, roc_curve,RocCurveDisplay, f1_score,
                             accuracy_score, classification_report)

## Page configuration
st.set_page_config(page_title='Fall Prediction App', layout='wide',
                  initial_sidebar_state='expanded')

## Title of the app
st.title('Interactive Fall Prediction Dashboard')

## Load dataset
df = pd.read_csv('Data/ml_df.csv')

## Remove Unnamed: 0 column
df = df.drop(columns='Unnamed: 0')

## Change first letter in each column to upper case
df.columns = df.columns.str.capitalize()

st.write(df)

## Input widgets for features
st.sidebar.subheader('Model Input Features')
distance = st.sidebar.slider('Distance (cm)', 0, 100, 28)
pressure = st.sidebar.slider('Pressure', 0, 1, 0)
hrv = st.sidebar.slider('Hrv (bpm)', 50, 200, 95)
blood_sugar = st.sidebar.slider('Blood sugar level (mg/dl)', 10, 200, 73)
spo2 = st.sidebar.slider('Spo2', 60, 100, 83)
accelerometer = st.sidebar.slider('Accelerometer', 0, 1, 1)

# Prepare a single observation for prediction
feature_vector = pd.DataFrame([[distance, pressure, hrv, blood_sugar, spo2, accelerometer]],
                              columns=['Distance (cm)', 'Pressure', 'Hrv (bpm)',
                                       'Blood sugar level (mg/dl)', 'Spo2', 'Accelerometer'])

## Split X and y
X = df.drop(columns=['Decision'])
y = df['Decision']

## Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## Model Preprocessing
## Column Transformers
scaler = StandardScaler()

## Column selectors
num_col = make_column_selector(dtype_include='number')

## Tuples for pipeline
num_tuple = (scaler, num_col)

## Preprocessor object
prepocessor = make_column_transformer(num_tuple, verbose_feature_names_out=False)

## Instantiate Model
rf = RandomForestClassifier()

## Model pipeline
rf_pipe = make_pipeline(prepocessor, rf)

## Fit Model
rf_pipe.fit(X_train, y_train)

# Define class mapping
class_mapping = {
    0: 'No Fall Detected',
    1: 'Stumble Detected',
    2: 'Fall Detected'
}

# Prediction button
if st.sidebar.button('Run Prediction'):
    prediction = rf_pipe.predict(feature_vector)[0]
    
    # Use the mapping to convert the numeric prediction to a meaningful label
    prediction_label = class_mapping[prediction]
    
    # Modify the output to reflect the class names or IDs relevant to your problem
    st.sidebar.write(f'Prediction: {prediction_label}')














