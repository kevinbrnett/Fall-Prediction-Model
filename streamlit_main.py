import streamlit as st
import time
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
shap.initjs()

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier


## Page configuration
st.set_page_config(page_title='Fall Prediction App',
                  initial_sidebar_state='expanded', layout='wide')

## Initialize session state
'session state object', st.session_state

                        ### FUNCTIONS ###
    
# Define a function to prepare data, train the model, and make a prediction
def make_prediction(distance, pressure, hrv, blood_sugar, spo2, accelerometer, df):
    # Prepare the observation for prediction
    feature_vector = pd.DataFrame([[distance, pressure, hrv, blood_sugar, spo2,
                                    accelerometer]], columns=['Distance (cm)', 'Pressure',
                                                              'Heart Rate (bpm)',
                                                              'Blood Sugar Level (mg/dL)',
                                                              'SpO2 (%)', 'Accelerometer'])

    # Split X and y
    X = df.drop(columns=['Decision'])
    y = df['Decision']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,
                                                        random_state=42)

    # Model preprocessing and pipeline
    scaler = StandardScaler()
    num_col = make_column_selector(dtype_include='number')
    num_tuple = (scaler, num_col)
    preprocessor = make_column_transformer(num_tuple, verbose_feature_names_out=False)
    rf = RandomForestClassifier(n_estimators=10)
    rf_pipe = make_pipeline(preprocessor, rf)

    # Fit the model
    rf_pipe.fit(X_train, y_train)

    # Make a prediction with the trained model
    prediction = rf_pipe.predict(feature_vector)
    prediction_proba = rf_pipe.predict_proba(feature_vector)

    # Define class mapping
    class_mapping = {
        0: 'No fall risk',
        1: 'Risk of stumbling',
        2: 'Risk of falling'
    }

    # Map confidence scores and prediction labels to corresponding class
    confidence_scores = {class_mapping[i]: prob for i, prob in
                         enumerate(prediction_proba[0])}
    prediction_label = class_mapping[prediction[0]]
    
    # Return prediction labels and confidence scores
    return prediction_label, confidence_scores

## Load data
df = pd.read_csv('Data/ml_df.csv')

## Remove Unnamed: 0 column
df.drop(columns='Unnamed: 0', inplace=True)

## Reformat column headers and target values for clarity
df.rename(columns={'distance (cm)':'Distance (cm)',
                   'pressure':'Pressure',
                   'hrv (bpm)':'Heart Rate (bpm)',
                   'blood sugar level (mg/dL)':'Blood Sugar Level (mg/dL)',
                   'spo2':'SpO2 (%)',
                   'accelerometer':'Accelerometer',
                   'decision':'Decision'
                  }, inplace=True)

                                ### SIDEBAR ###
## Input widgets for features
st.sidebar.subheader('Input Data:')

## Distance with slider and number input
distance = st.sidebar.slider('Distance (cm):', min(df['Distance (cm)']), 
                             max(df['Distance (cm)'].round()), key='distance_slider')

## Pressure with slider and number input
pressure = st.sidebar.slider('Pressure:', 0, 2, key='pressure_slider')

## Heart Rate with slider and number input
hrv = st.sidebar.slider('Heart Rate (bpm):', min(df['Heart Rate (bpm)']), 
                        max(df['Heart Rate (bpm)'].round()), key='hrv_slider')

## Blood Sugar Level with slider and number input
blood_sugar = st.sidebar.slider('Blood Sugar Level (mg/dL):', 
                                min(df['Blood Sugar Level (mg/dL)']),
                                max(df['Blood Sugar Level (mg/dL)'].round()),
                                key='blood_sugar_slider')

## SpO2 with slider and number input
spo2 = st.sidebar.slider('SpO2 (%):', min(df['SpO2 (%)']),
                         max(df['SpO2 (%)'].round()), key='spo2_slider')

## Accelerometer with slider and number input
accelerometer = st.sidebar.slider('Accelerometer:', 0, 1, key='accelerometer_slider')

                       ### INITIALIZE SLIDER SESSION STATES ###
if 'distance_slider' not in st.session_state:
    st.session_state['distance_slider'] = distance

if 'pressure_slider' not in st.session_state:
    st.session_state['pressure_slider'] = pressure

if 'hrv_slider' not in st.session_state:
    st.session_state['hrv_slider'] = hrv
    
if 'blood_sugar_slider' not in st.session_state:
    st.session_state['blood_sugar_slider'] = blood_glucose

if 'spo2_slider' not in st.session_state:
    st.session_state['spo2_slider'] = spo2
    
if 'accelerometer_slider' not in st.session_state:
    st.session_state['accelerometer_slider'] = accelerometer
    


    
    

                             ### MODEL PREDICTION BUTTON ###


if st.sidebar.button('Run Prediction'):
    prediction_label, confidence_scores = make_prediction(distance, pressure, hrv,
                                                          blood_sugar, spo2, accelerometer,
                                                          df)  
    max_confidence_label = max(confidence_scores, key=confidence_scores.get)
    max_confidence_score = confidence_scores[max_confidence_label]
    st.session_state['predict_button_clicked'] = True
    
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(height=370, border=True):
      
        # Initialize or use a placeholder for the prediction result
        prediction_placeholder = st.empty()
        prediction_placeholder.text('Input values in the sidebar and click the run prediction button to calculate')

        # Check if the prediction button was clicked
        if 'predict_button_clicked' in st.session_state:
            # Display a loading message or similar feedback
            prediction_placeholder.text('Calculating prediction...')

            # Simulate a delay for prediction processing
            time.sleep(2)
            
            ## Show output
            prediction_placeholder.write(prediction_label)
            
with col2:
    with st.container(height=370, border=True):
        # Initialize or use a placeholder for the prediction result
        prediction_placeholder = st.empty()
        prediction_placeholder.text('Input values in the sidebar and click the run prediction button to calculate')

        # Check if the prediction button was clicked
        if 'predict_button_clicked' in st.session_state:
            
            ## Show output
            prediction_placeholder.write(max_confidence_score)
        
with col3:
    with st.container(height=370, border=True):
            # Initialize or use a placeholder for the prediction result
        prediction_placeholder = st.empty()
        prediction_placeholder.text('Input values in the sidebar and click the run prediction button to calculate')

        # Check if the prediction button was clicked
        if 'predict_button_clicked' in st.session_state:
            
            ## Show output
            prediction_placeholder.write(prediction_label)




