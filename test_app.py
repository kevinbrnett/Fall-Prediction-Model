import time
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
shap.initjs()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline

## Page configuration
st.set_page_config(page_title='Fall Prediction App', initial_sidebar_state='expanded', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

## Button to make predictions
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = 0
    
## Load Data
data = pd.read_csv('Data/ml_df.csv')
data.drop(columns='Unnamed: 0', inplace=True)

X = data.drop(columns='decision')
y = data['decision']

## Sidebar
st.sidebar.header('Specify Input Parameters')

distance = st.sidebar.slider('Distance (cm)', X.distance.min(), X.distance.max(), X.distance.mean(), key='distance')
pressure = st.sidebar.slider('Pressure', 0, 2, 1, key='pressure')
hrv = st.sidebar.slider('Heart Rate (bpm)', X.hrv.min(), X.hrv.max(), X.hrv.mean(), key='hrv')
blood_sugar = st.sidebar.slider('Blood Sugar Level (mg/dL)', X.blood_sugar.min(), X.blood_sugar.max(), X.blood_sugar.mean(), key='blood_sugar')
spo2 = st.sidebar.slider('SpO2 (%)', X.spo2.min(), X.spo2.max(), X.spo2.mean(), key='spo2')
accelerometer = st.sidebar.slider('Accelerometer', 0, 1, 1, key='accelerometer')


def user_input_features():

    data = {'distance': distance,
            'pressure': pressure,
            'hrv': hrv,
            'blood_sugar': blood_sugar,
            'spo2': spo2,
            'accelerometer': accelerometer}
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

## Print specified input parameters
st.header('Specified Input parameters')
st.write(user_input)
st.divider()

def interpret_prediction(prediction):
    if prediction == 0:
        return "No Fall Risk"
    elif prediction == 1:
        return "Risk of Stumbling"
    elif prediction == 2:
        return "Risk of Falling"
    else:
        return "Invalid Prediction"  # Handle unexpected values

def prediction_callback():
    st.session_state.button_clicked = 1


    # Formatting the probability output for clarity
    proba_df = pd.DataFrame(prediction_probability, columns=model.classes_)
    st.write('Prediction Probabilities:')
    st.write(proba_df)

## Build Random Forest Model
rf = RandomForestClassifier(class_weight='balanced', max_depth=25, n_estimators=50, random_state=42)

## Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## Model preprocessing and pipeline
scaler = StandardScaler()
num_col = make_column_selector(dtype_include='number')
num_tuple = (scaler, num_col)
preprocessor = make_column_transformer(num_tuple, verbose_feature_names_out=False)

model = make_pipeline(preprocessor, rf)

## Fit the model
model.fit(X_train, y_train)

prediction = model.predict(user_input)
prediction_probability = model.predict_proba(user_input)

'before button clicked', st.session_state

prediction_button = st.sidebar.button('Run Prediction', on_click=prediction_callback)

if prediction_button:
    'after clicking button', st.session_state
    
    
    
    
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.subheader('Model Prediction')
        
        if st.session_state.button_clicked == 0:
            # Create a placeholder
            placeholder = st.empty()
            placeholder.text("Select prediction parameters and click 'Run Prediction' button")

        if st.session_state.button_clicked == 1:
            with st.spinner('Calculating Prediction...'):
                time.sleep(2)

            st.write(interpret_prediction(prediction))
