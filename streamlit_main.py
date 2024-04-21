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

    # Access the RandomForestClassifier from the pipeline
    rf_classifier = rf_pipe.named_steps['randomforestclassifier']

    # Transform the test set
    X_test_df = pd.DataFrame(rf_pipe[:-1].transform(X_test),
                             columns=feature_names,
                             index=X_test.index)

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test_df)
    
    
    # Return prediction labels and confidence scores
    return prediction_label, confidence_scores, shap_values
                            
    
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

                                ### INFORMATIONAL TABS ###

tab1, tab2, tab3, tab4 = st.tabs(['About', 'What is Fall?', 'Importance', 'Risk Factors'])

with tab1:
    with st.container(border=False):
        st.header('About this App')
            
      
        st.write('The purpose of this app is to make new predictions for an individual\'s fall risk based on six variables. ')
        st.write('In order to make predictions fill in the variables in the side bar and click the run prediction button. Prediction results will appear in the model prediction tile after calculating.')

with tab2:
    with st.container(border=False):
        st.header('Definition of a Fall')
        st.write('A fall is defined as a “sudden, not intentional, and unexpected movement from orthostatic position, from seat to position, or from clinical position”.')
    
with tab3:
    with st.container(border=False):
        st.header('Importance of Falls in Hospital')

with tab4:
    with st.container(border=False):
        st.header('Who is at Risk for Falling?')
        risk_factors = [ "Older than 85","Weight","History of falls",
                        "Mobility problems","Use of assistivedevices",
                        "Medications","Mental status","Incontinence","Vision impairment"]
    # Creating a markdown string for the list
        markdown_list = "\n".join(f"- {factor}" for factor in risk_factors)
        st.markdown(markdown_list)

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
    prediction_label, confidence_scores, shap_values = make_prediction(distance, pressure, hrv,
                                                          blood_sugar, spo2, accelerometer, df)  
    max_confidence_label = max(confidence_scores, key=confidence_scores.get)
    max_confidence_score = confidence_scores[max_confidence_label]
    st.session_state['predict_button_clicked'] = True
            
                                 ### EDA HISTOGRAM ###

with st.container(border=True):
    st.header('Distribution of the Data')
    
    col1, col2 = st.columns([0.6,0.4])
    
    with col1:
        # Melt the DataFrame
        long_df = pd.melt(df, value_vars=['Distance (cm)', 'Heart Rate (bpm)',
                         'Blood Sugar Level (mg/dL)','SpO2 (%)'],
                          var_name='Variable', value_name='Value')

        # Create the histogram
        fig = px.histogram(long_df, x='Value', color='Variable', barmode='overlay')

        # Update the layout
        fig.update_layout(
            xaxis_title_text='Value', 
            yaxis_title_text='Count', 
            legend_title_text='Variables',
            width=800,
            height=800)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

                                        ### EDA COUNT PLOT ###
    with col2:
        # Mapping from numeric to descriptive categories
        category_mapping = {
            0: 'No Fall',
            1: 'Stumbled',
            2: 'Fall'
        }

        # Apply the mapping to each category column
        df_mapped = df.replace(category_mapping)


        # Melt the DataFrame to long format
        long_df = pd.melt(df_mapped, value_vars=['Pressure', 'Accelerometer'],
                          var_name='Variable', value_name='Category')

        # Create the count plot
        fig = px.histogram(long_df, x='Category', color='Variable', barmode='overlay', text_auto=True)

        # Update the layout
        fig.update_layout(
            xaxis_title_text='Decision',
            yaxis_title_text='Count',
            legend_title_text='Variables',
            barmode='overlay',
            width=800,
            height=800)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
                                      ### CORRELATION MATRIX ###
# Compute the correlation matrix
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
corr_masked = corr.where(~mask, np.nan)
custom_color_scale = [
    [0.0, '#216CD3'],  # Start at 0
    [0.5, 'white'],  # Transition through white at 0.5
    [1.0, '#FF7B13']   # End at blue at 1
]

with st.container(border=True):
    st.header('Correlation in the Data')
    
    # Create the heatmap with values displayed
    fig = go.Figure(data=go.Heatmap(
        z=corr_masked,
        x=corr.columns,
        y=corr.index,
        colorscale=custom_color_scale,
        hoverongaps=False,
        text=np.round(corr_masked.to_numpy(), decimals=2),  # Include rounded values as text
        texttemplate="%{text}",  # Use text values as text template
        showscale=True,  # Show the color scale bar
        textfont=dict(size=16)  # Increase the text font size
    ))

    # Update layout with titles and axis labels, specifying font sizes
    fig.update_layout(
    xaxis=dict(
        tickmode='linear',
        title='Features',  # x-axis title
        title_font={'size': 20},  # Font size for x-axis title
        tickfont={'size': 16}  # Font size for x-axis tick labels
    ),
    yaxis=dict(
        tickmode='linear',
        title='Features',  # y-axis title
        title_font={'size': 20},  # Font size for y-axis title
        tickfont={'size': 16}  # Font size for y-axis tick labels
    ),
    width=1000,
    height=800
    )

    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))


    # Display the heatmap in Streamlit
    st.plotly_chart(fig, use_container_width=False)

                                    ### PHYSIOLOGIC BOX PLOTS ###              

# Define custom color palette
colors = ['#FF7B13', '#E32A2A', '#216CD3']
cols = ['Heart Rate (bpm)', 'Blood Sugar Level (mg/dL)', 'SpO2 (%)']

with st.container (border=True):
    st.header('Trends in Physiologic Variables') 
    
    fig = make_subplots(rows=1, cols=3)

    for i, col in enumerate(cols, 1):
        fig.add_trace(
            go.Box(x=df['Decision'], y=df[col], name=col,
                   marker_color=colors[i % len(colors)]),
            row=1, col=i
        )

        mean_val = df[col].mean()
        fig.add_trace(
            go.Scatter(x=[min(df['Decision']), max(df['Decision'])], y=[mean_val, mean_val],
                       mode='lines', line=dict(color='black', dash='dash'), name='Average'),
            row=1, col=i
        )

    fig.update_layout(
        height=800,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

                            ### MACHINE LEARNING TILES ###
    
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

        # Check if the prediction button was clicked
        if 'predict_button_clicked' in st.session_state:
            
            ## Show output
            prediction_placeholder.write(max_confidence_score)
        
with col3:
    with st.container(height=370, border=True):
            # Initialize or use a placeholder for the prediction result
        prediction_placeholder = st.empty()

        # Check if the prediction button was clicked
        if 'predict_button_clicked' in st.session_state:
            
            ## Show output
            prediction_placeholder.write('tbd')
            
            
# Assuming you want to analyze the first (and only) instance in the input
instance_index = 0

# Determine the most important feature for each class
important_features = {}
for class_index in range(3):  # Adjust based on the number of classes
    class_shap_values = shap_values[class_index][instance_index]
    most_important_feature_index = np.argmax(np.abs(class_shap_values))
    most_important_feature = X_test_df.columns[most_important_feature_index]
    important_features[f'Class {class_index}'] = most_important_feature

# Display the results
st.write("Most important features for each class:")
for class_label, feature in important_features.items():
    st.write(f"{class_label}: {feature}")
            





