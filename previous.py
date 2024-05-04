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
st.set_page_config(page_title='Fall Prediction App', initial_sidebar_state='expanded', layout='wide')

                        ### APP TITLE AND LOGO ###
    
image_path = 'Visuals/logo.png'

col1, col2 = st.columns([0.25,0.75])
with col1:
    st.image(image_path, use_column_width=True)

with col2:
    ## Title of the app
    st.title('Interactive Fall Prediction Dashboard')
    
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
    
    ## Calculate SHAP values
    # Feature names extracted from the training set
    feature_names = X_train.columns

    
    # Transform the test set
    X_test_df = pd.DataFrame(rf_pipe[:-1].transform(X_test),
                             columns=feature_names,
                             index=X_test.index)
    
    instance_index = 0
    instance_data = X_test_df.iloc[[instance_index]]
    predictions = rf_pipe.predict(instance_data)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
               

    # Access the RandomForestClassifier from the pipeline
    rf_classifier = rf_pipe.named_steps['randomforestclassifier']

    # Transform the test set
    X_test_df = pd.DataFrame(rf_pipe[:-1].transform(X_test),
                             columns=feature_names,
                             index=X_test.index)

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test_df)
    shap_values = explainer.shap_values(instance_data)
    
    
    # Return prediction labels and confidence scores
    return prediction_label, confidence_scores, shap_values
    return prediction_label, confidence_scores, shap_values, X_test_df
                            
    
## Load data


@@ -209,7 +210,7 @@ if 'accelerometer_slider' not in st.session_state:


if st.sidebar.button('Run Prediction'):
    prediction_label, confidence_scores, shap_values = make_prediction(distance, pressure, hrv,
    prediction_label, confidence_scores, shap_values, X_test_df = make_prediction(distance, pressure, hrv,
                                                          blood_sugar, spo2, accelerometer, df)  
    max_confidence_label = max(confidence_scores, key=confidence_scores.get)
    max_confidence_score = confidence_scores[max_confidence_label]


@@ -404,23 +405,11 @@ with col3:
            ## Show output
            prediction_placeholder.write('tbd')
            
            
# Assuming you want to analyze the first (and only) instance in the input
instance_index = 0

# Determine the most important feature for each class
important_features = {}
for class_index in range(3):  # Adjust based on the number of classes
    class_shap_values = shap_values[class_index][instance_index]
    most_important_feature_index = np.argmax(np.abs(class_shap_values))
    # Find the most important feature for the predicted class
    most_important_feature_index = np.argmax(np.abs(shap_values[predicted_class_index][0]))
    most_important_feature = X_test_df.columns[most_important_feature_index]
    important_features[f'Class {class_index}'] = most_important_feature
    st.write(f'Most Important Feature for Predicted Class: {most_important_feature}')

# Display the results
st.write("Most important features for each class:")
for class_label, feature in important_features.items():
    st.write(f"{class_label}: {feature}")