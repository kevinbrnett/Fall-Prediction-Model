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

## CSS for custom font size
st.markdown("""
    <style>
    h1 {
        font-size: 140px;
    }
    h2 {
        font-size: 80px;
    }
     h3 {
        font-size: 70px;
    }
    </style>
    """, unsafe_allow_html=True)

image_path = 'Visuals/logo.png'

col1, col2 = st.columns([0.25,0.75])

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
        st.write('The purpose of this app is to make new predictions for an individual\'s fall risk based on six variables. ')
        st.write('**Data Dictionary** Distance: distance from nearest object, Pressure: if pressure threshold was met or not, Accelerometer: if accelerometer threshold was met or not, Blood sugar: blood glucose level, Heart rate: heart rate in beats per minute, Spo2: oxygen saturation percent, Decision: target variable (no fall, stumble, fall)')
        st.write('In order to make predictions fill in the variables in the side bar and click the run prediction button. Prediction results will appear in the model prediction tile after calculating.')

with tab2:
    with st.container(border=True):
        st.header('Definition of a Fall')
        st.write('A fall is defined as a “sudden, not intentional, and unexpected movement from orthostatic position, from seat to position, or from clinical position”.')
    
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
                    <h3>{text}</h3>
                </div>
            """, unsafe_allow_html=True)

header1 = 'Fall Rate'            
text1 = f'{fall_rate} %'
header2 = 'placeholder'
text2 = 'what to put'
header3 = 'placeholder'
text3 = 'what to put'

# Create containers with centered headers and texts
create_container(col1, header1, text1)
create_container(col2, header2, text2)
create_container(col3, header3, text3)

                                    ### PREDICTION SIDEBAR ###

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
                            
                                ### MACHINE LEARNING MODEL ###

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
    confidence_scores = {class_mapping[i]: prob for i, prob in enumerate(prediction_proba[0])}
    prediction_label = class_mapping[prediction[0]]
    
    # Return prediction labels and confidence scores
    return prediction_label, confidence_scores


                                            ### MODEL PREDICTION BUTTON ###

## Prediction button
if st.sidebar.button('Run Prediction'):
    prediction_label, confidence_scores = make_prediction(distance, pressure, hrv, blood_sugar, spo2,
                                                          accelerometer, df)
    st.session_state['prediction_label'] = prediction_label
    st.session_state['confidence_scores'] = confidence_scores
    st.session_state['predict_button_clicked'] = True

        
st.header('Exploring the Distribution of the Data')

                                        ### EDA HISTOGRAM ###
    
col1, col2 = st.columns([0.6,0.4])

with col1:
    with st.container(border=True):
        # Melt the DataFrame
        long_df = pd.melt(df, value_vars=['Distance (cm)', 'Heart Rate (bpm)', 'Blood Sugar Level (mg/dL)',
                                          'SpO2 (%)'], var_name='Variable', value_name='Value')

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

# Mapping from numeric to descriptive categories
category_mapping = {
    0: 'No Fall',
    1: 'Stumbled',
    2: 'Fall'
}

# Apply the mapping to each category column
df_mapped = df.replace(category_mapping)

with col2:
    with st.container(border=True):
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
            
st.header('Analyzing Trends in Physiologic Variables')            

# Define custom color palette
colors = ['#FF7B13', '#E32A2A', '#216CD3']
cols = ['Heart Rate (bpm)', 'Blood Sugar Level (mg/dL)', 'SpO2 (%)']

with st.container (border=True):
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

# ## Access the RandomForestClassifier from the pipeline
# rf_classifier = rf_pipe.named_steps['randomforestclassifier']

# ## Create explainer
# explainer = shap.TreeExplainer(rf_classifier)

# ## Calculate SHAP values
# shap_values = explainer.shap_values(X_test)

# ## Dropdown to select the class
# st.subheader('Feature Importances by Class')

# class_option = st.selectbox('Select a class:', range(3))
# st.subheader(f'SHAP Summary Plot for Class {class_option}')
    
# ## SHAP summary plot
# fig, ax = plt.subplots()
# shap.summary_plot(shap_values[class_option], X_test, plot_type='bar', show=False)
# st.pyplot(fig)

                                        ### MACHINE LEARNING TILES ###
        
col1, col2, col3 = st.columns(3)


with col1:
    with st.container(height=370, border=True):
        st.markdown(f"""
                <style>
                .centered-text {{
                    text-align: center;
                }}
                </style>
                <div class="centered-text">
                    <h2>Model Prediction</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Initialize or use a placeholder for the prediction result
        prediction_placeholder = st.empty()
        prediction_placeholder.text('Input values in the sidebar and click the run prediction button to calculate\na prediction')
        
         # Define custom style
        custom_style = """
                <style>
                    .prediction {
                        font-size: 80px;
                        color: #447eeb; 
                        font-family: Helvetica, sans-serif;
                        text-align: center;
                        word-wrap: break-word;
                        overflow-wrap: break-word;
                        margin: 10px;
                    }
                </style>
                """
        # Apply the custom style to the app
        st.markdown(custom_style, unsafe_allow_html=True)
        
        # Check if the prediction button was clicked
        if 'predict_button_clicked' in st.session_state:
            # Display a loading message or similar feedback
            prediction_placeholder.text('Calculating prediction...')

            # Simulate a delay for prediction processing
            time.sleep(2) 

            # Update the placeholder with the prediction result using custom styling
            prediction_text = f"<div class='prediction'>{st.session_state['prediction_label']}</div>"
            prediction_placeholder.markdown(prediction_text, unsafe_allow_html=True)

            # Reset the button click state if desired
            st.session_state.predict_button_clicked = False
                 
            
with col2:
    with st.container(height=370, border=True):
        st.markdown("""
                    <style>
                    .centered-text {
                        text-align: center;
                    }
                    </style>
                    <div class="centered-text">
                    <h2>Prediction Score</h2>
                    </div>
                """, unsafe_allow_html=True)
        
         # Define custom style
        custom_style = """
                <style>
                    .confidence {
                        font-size: 80px;
                        color: #447eeb; 
                        font-family: Helvetica, sans-serif;
                        text-align: center;
                        word-wrap: break-word;
                        overflow-wrap: break-word;
                        margin: 10px;
                    }
                </style>
                """
        # Apply the custom style to the app
        st.markdown(custom_style, unsafe_allow_html=True)
        
        # Logic to display only the confidence score for the predicted class
        if 'predict_button_clicked' in st.session_state:
            if 'confidence_scores' in st.session_state and 'prediction_label' in st.session_state:
                predicted_class_confidence = st.session_state['confidence_scores'][st.session_state['prediction_label']]

                # Display the confidence score with custom styling
                confidence_html = f"<div class='confidence'>{predicted_class_confidence:.2%}</div>"
                st.markdown(confidence_html, unsafe_allow_html=True)
        
        # Reset the button click state if desired
        st.session_state.predict_button_clicked = False
        
with col3:
    with st.container(height=370, border=True):
        st.markdown(f"""
                    <style>
                    .centered-text {{
                        text-align: center;
                    }}
                    </style>
                    <div class="centered-text">
                    <h2>Most Important Factor</h2>
                    </div>
                """, unsafe_allow_html=True)




