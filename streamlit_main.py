import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
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

## CSS for custom font size
st.markdown("""
    <style>
    h1 {
        font-size: 140px;
    }
    h2 {
        font-size: 80px;
    }
    </style>
    """, unsafe_allow_html=True)

image_path = 'Visuals/logo.png'

col1, col2 = st.columns([0.2,0.8])

with col1:
    st.image(image_path, use_column_width=True)

with col2:
    ## Title of the app
    st.title('Interactive Fall Prediction Dashboard')

## Informational tabs about the app
tab1, tab2, tab3, tab4 = st.tabs(['About', 'What is Fall?', 'Importance', 'Risk Factors'])

with tab1:
    with st.container(border=True):
        st.header('About this App')
        st.write('this is a placeholder for information on the app. slkdjfsladjflaksjdflkajsldjflaasdflkasjdflasjdflasjdflkajsdflkajsdlfjaslkdfjlaskdjfalskdjflasdjkflasdjkflasdjflasjkdlasdjflaskdj')

with tab2:
    with st.container(border=True):
        st.header('Definition of a Fall')
    
with tab3:
    with st.container(border=True):
        st.header('Importance of Falls in Hospital')

with tab4:
    with st.container(border=True):
        st.header('Who is at Risk for Falling?')
    
## Load dataset
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

## Calculations for fall rate percent
num_falls = df['Decision'].loc[df['Decision'] == 2].sum()
num_other = df['Decision'].loc[df['Decision'] != 2].sum()
total = num_falls + num_other
fall_rate = ((num_falls / total) *100).round(2)

## Statistic tiles
## Row with 3 coulmn tiles
col1, col2, col3 = st.columns(3)

## Function to create a container with a centered header and text
def create_container(column, header, text):
    with column:
        with st.container(height=280, border=True):
            st.markdown(f"""
                <style>
                .centered-text {{
                    text-align: center;
                }}
                </style>
                <div class="centered-text">
                    <h2>{header}</h2>
                    <h2>{text}</h2>
                </div>
            """, unsafe_allow_html=True)

header1 = 'Fall Rate'            
text1 = f'{fall_rate} %'

# Create containers with centered headers and texts
create_container(col1, header1, text1)
create_container(col2, header1, text1)
create_container(col3, header1, text1)

## Input widgets for features
st.sidebar.subheader('Variables for Model Prediction')

## Distance with slider and number input
distance = st.sidebar.slider('Distance (cm):', min(df['Distance (cm)']), 
                             max(df['Distance (cm)'].round()))
distance = st.sidebar.number_input('Set exact Distance (cm):', 
                                   min_value=float(min(df['Distance (cm)'])), 
                                   max_value=float(max(df['Distance (cm)'].round())), 
                                   value=float(distance))

## Pressure with slider and number input
pressure = st.sidebar.slider('Pressure:', 0, 2)
pressure = st.sidebar.number_input('Set exact Pressure:', 
                                   min_value=0, max_value=2, 
                                   value=int(pressure))

## Heart Rate with slider and number input
hrv = st.sidebar.slider('Heart Rate (bpm):', min(df['Heart Rate (bpm)']), 
                        max(df['Heart Rate (bpm)'].round()))
hrv = st.sidebar.number_input('Set exact Heart Rate (bpm):', 
                              min_value=float(min(df['Heart Rate (bpm)'])), 
                              max_value=float(max(df['Heart Rate (bpm)'].round())), 
                              value=float(hrv))

## Blood Sugar Level with slider and number input
blood_sugar = st.sidebar.slider('Blood Sugar Level (mg/dL):', 
                                min(df['Blood Sugar Level (mg/dL)']),
                                max(df['Blood Sugar Level (mg/dL)'].round()))
blood_sugar = st.sidebar.number_input('Set exact Blood Sugar Level (mg/dL):', 
                                      min_value=float(min(df['Blood Sugar Level (mg/dL)'])),
                                      max_value=float(max(df['Blood Sugar Level (mg/dL)'].round())),
                                      value=float(blood_sugar))

## SpO2 with slider and number input
spo2 = st.sidebar.slider('SpO2 (%):', min(df['SpO2 (%)']), max(df['SpO2 (%)'].round()))
spo2 = st.sidebar.number_input('Set exact SpO2 (%):', 
                               min_value=float(min(df['SpO2 (%)'])),
                               max_value=float(max(df['SpO2 (%)'].round())), 
                               value=float(spo2))

## Accelerometer with slider and number input
accelerometer = st.sidebar.slider('Accelerometer:', 0, 1)
accelerometer = st.sidebar.number_input('Set exact Accelerometer:', 
                                        min_value=0, max_value=1, 
                                        value=int(accelerometer))
                            
# Prepare a single observation for prediction
feature_vector = pd.DataFrame([[distance, pressure, hrv, blood_sugar, spo2, accelerometer]],
                              columns=['Distance (cm)', 'Pressure', 'Heart Rate (bpm)',
                                       'Blood Sugar Level (mg/dL)', 'SpO2 (%)', 'Accelerometer'])

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
rf = RandomForestClassifier(n_estimators=10)

## Model pipeline
rf_pipe = make_pipeline(prepocessor, rf)

## Fit Model
rf_pipe.fit(X_train, y_train)

## Define class mapping
class_mapping = {
    0: 'No Fall Detected',
    1: 'Stumble Detected',
    2: 'Fall Detected'
}

## Prediction button
if st.sidebar.button('Run Prediction'):
    prediction = rf_pipe.predict(feature_vector)[0]
    
    ## Use the mapping to convert the numeric prediction to a meaningful label
    prediction_label = class_mapping[prediction]
    
    ## Modify the output to reflect the class names or IDs relevant to your problem
    st.sidebar.write(f'Prediction: {prediction_label}')

st.subheader('Data Visualization for Continuous Features')

# Determine continuous features based on a heuristic
# Consider a feature continuous if it has more than 10 unique values
continuous_features = [feature for feature in df.columns if df[feature].nunique() > 10]

# If no continuous features are found, display a message
if not continuous_features:
    st.write("No continuous features available.")
else:
    # Feature selection dropdown
    selected_feature = st.selectbox('Select a continuous feature:', continuous_features)

    # Create two columns for the histogram and the boxplot
    col1, col2 = st.columns(2)

    # Display histogram in the first column
    with col1:
        st.subheader(f'Histogram of {selected_feature}')
        fig1, ax1 = plt.subplots()
        ax1.hist(df[selected_feature], bins=20, edgecolor='black')
        ax1.set_xlabel(selected_feature)
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)

    # Display boxplot in the second column
    with col2:
        st.subheader(f'Boxplot of {selected_feature}')
        fig2, ax2 = plt.subplots()
        ax2.boxplot(df[selected_feature], vert=False)
        ax2.set_xlabel(selected_feature)
        st.pyplot(fig2)

## Access the RandomForestClassifier from the pipeline
rf_classifier = rf_pipe.named_steps['randomforestclassifier']

## Create explainer
explainer = shap.TreeExplainer(rf_classifier)

## Calculate SHAP values
shap_values = explainer.shap_values(X_test)

## Dropdown to select the class
st.subheader('Feature Importances by Class')

class_option = st.selectbox('Select a class:', range(3))
st.subheader(f'SHAP Summary Plot for Class {class_option}')
    
## SHAP summary plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values[class_option], X_test, plot_type='bar', show=False)
st.pyplot(fig)






