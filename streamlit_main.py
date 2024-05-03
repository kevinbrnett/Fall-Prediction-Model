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

st.set_option('deprecation.showPyplotGlobalUse', False)

## Load Data
data = pd.read_csv('Data/ml_df.csv')
data.drop(columns='Unnamed: 0', inplace=True)

X = data.drop(columns='decision')
y = data['decision']

## Sidebar
st.sidebar.header('Specify Input Parameters')

def user_input_features():

    distance = st.sidebar.slider('Distance (cm)', X.distance.min(), X.distance.max(), X.distance.mean())
    pressure = st.sidebar.slider('Pressure', 0, 2, 1)
    hrv = st.sidebar.slider('Heart Rate (bpm)', X.hrv.min(), X.hrv.max(), X.hrv.mean())
    blood_sugar = st.sidebar.slider('Blood Sugar Level (mg/dL)', X.blood_sugar.min(), X.blood_sugar.max(), X.blood_sugar.mean())
    spo2 = st.sidebar.slider('SpO2 (%)', X.spo2.min(), X.spo2.max(), X.spo2.mean())
    accelerometer = st.sidebar.slider('Accelerometer', 0, 1, 1)


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

## Button to make predictions
if st.sidebar.button('Predict'):
    prediction = model.predict(user_input)
    prediction_probability = model.predict_proba(user_input)
    st.write(f'Prediction: Class {prediction[0]}')
    
    # Formatting the probability output for clarity
    proba_df = pd.DataFrame(prediction_probability, columns=model.classes_)
    st.write('Prediction Probabilities:')
    st.write(proba_df)

## Calculating SHAP values
explainer = shap.TreeExplainer(model[-1], X_train)
shap_values = explainer(user_input)

fig = shap.plots.waterfall(shap_values[0, :, 0])
st.pyplot(fig)

feature_names = model[:-1].get_feature_names_out()
feature_importance = pd.Series(model[-1].feature_importances_, index=feature_names)

# Create a DataFrame from the series for easier handling with Plotly
feature_importance_df = pd.DataFrame({'Feature': feature_importance.index, 'Importance': feature_importance.values})

# Sort the DataFrame by importance
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Visualize using Plotly Express - showing all features or just top 4 if needed
fig = px.bar(sorted_feature_importance_df, x='Importance', y='Feature', orientation='h', 
             title='Average Feature Importances')
fig.update_layout(xaxis_title='Feature Importance Value', yaxis_title='Feature Name')

# Show the plot
st.plotly_chart(fig)
