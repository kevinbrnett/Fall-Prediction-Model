import time
import streamlit as st
import pandas as pd
import numpy as np
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
from sklearn.metrics import classification_report
from imblearn.pipeline import make_pipeline

## Page configuration
st.set_page_config(page_title='Fall Prediction App', initial_sidebar_state='expanded', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

# ## Initialize session state
# 'session state object', st.session_state

## Button to make predictions
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

## Load Data
data = pd.read_csv('Data/ml_df.csv')
data.drop(columns='Unnamed: 0', inplace=True)


X = data.drop(columns='decision')
y = data['decision']

## Sidebar
st.sidebar.header('Specify Input Parameters')

distance = st.sidebar.slider('**Distance (cm)**', X.distance.min(), X.distance.max(), X.distance.mean(), key='distance')
pressure = st.sidebar.slider('**Pressure**', 0, 2, 1, key='pressure')
hrv = st.sidebar.slider('**Heart Rate (bpm)**', X.hrv.min(), X.hrv.max(), X.hrv.mean(), key='hrv')
blood_sugar = st.sidebar.slider('**Blood Sugar Level (mg/dL)**', X.blood_sugar.min(), X.blood_sugar.max(), X.blood_sugar.mean(), key='blood_sugar')
spo2 = st.sidebar.slider('**SpO2 (%)**', X.spo2.min(), X.spo2.max(), X.spo2.mean(), key='spo2')
accelerometer = st.sidebar.slider('**Accelerometer**', 0, 1, 1, key='accelerometer')

                                                   ### INITIALIZE SLIDER SESSION STATES ###
if 'distance' not in st.session_state:
    st.session_state['distance'] = distance

if 'pressure' not in st.session_state:
    st.session_state['pressure'] = pressure

if 'hrv' not in st.session_state:
    st.session_state['hrv'] = hrv
    
if 'blood_sugar' not in st.session_state:
    st.session_state['blood_sugar'] = blood_sugar

if 'spo2' not in st.session_state:
    st.session_state['spo2'] = spo2
    
if 'accelerometer' not in st.session_state:
    st.session_state['accelerometer'] = accelerometer

def button_clicked():
    st.session_state.button_clicked = True

def interpret_prediction(prediction):
    if prediction == 0:
        return "<div class='container no-risk'><h2>Model Prediction</h2><p class='result-text'>No Fall Risk</p></div>"
    elif prediction == 1:
        return "<div class='container orange-text'><h2>Model Prediction</h2><p class='result-text'>Risk of Stumbling</p></div>"
    elif prediction == 2:
        return "<div class='container red-text'><h2>Model Prediction</h2><p class='result-text'>RISK OF FALLING</p></div>"
    else:
        return "<div class='container'><h2>Model Prediction</h2><p class='result-text'>Invalid Prediction</p></div>"  # Handle unexpected values

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

                                                  ### APP TITLE AND LOGO ###
    
image_path = 'Visuals/logo.png'

col1, col2, col3, col4 = st.columns([0.1,0.3,0.5,0.1])
    
with col2:
    st.image(image_path, width=450)

with col3:
    
    # Custom CSS
    custom_css = """
    <style>
    .custom-title {
        font-size: 120px; /* Adjust the size as needed */
        font-weight: bold;
        text-align: center;
        color: #333; /* Adjust the color as needed */
        margin-bottom: 20px; /* Adjust the margin as needed */
    }
    </style>
    """

    # Inject the custom CSS style
    st.markdown(custom_css, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="custom-title">Fall Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<h1 class="custom-title">Dashboard</h1>', unsafe_allow_html=True)
    
with col4:
    
     # CSS for positioning text at the bottom
        css = """
        <style>
        .container {
            position: relative;
            height: 500px;
            border: 1px solid black;
        }
        .bottom-text {
            position: absolute;
            bottom: 0;
            width: 100%;
            text-align: left;
            margin: 0;
        }
        </style>
        """

        # HTML content with the CSS class
        html_content = """
        <div class="container">
          <p class="bottom-text">Model Owner: Kevin Barnett, Data Scientist<br>Data Source: Hospital Database</p>
        </div>
        """

        # Display the HTML content with CSS in a Streamlit container
        st.markdown(css + html_content, unsafe_allow_html=True)
        
    
st.divider()
                                                   ### INFORMATIONAL SECTION ###
    
col1, col3 = st.columns([0.6,0.4])

with col1:
    with st.container(border=False):
        # Custom CSS
        custom_css = """
        <style>
        
        .custom-warning {
            font-size: 25px; /* Adjust the size as needed */
            font-weight: bold;
            text-align: center;
            background-color: #fff3cd;
            color: #B2612D; /* Adjust the color as needed */
            margin-bottom: 20px; /* Adjust the margin as needed */
            border-radius: 15px; /* Adjust border radius as needed */
            padding: 30px; /* Adjust padding as needed */
        }
        </style>
        """

        # Inject the custom CSS style
        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<p class="custom-warning">Due to suboptimal data the current model used in this app is not recommended to be used for real-world patient predictions</p>', unsafe_allow_html=True)
        
        

        # Custom CSS
        custom_css = """
        <style>
        .custom-about {
            font-size: 40px; /* Adjust the size as needed */
            font-weight: normal;
            color: #333; /* Adjust the color as needed */
            margin-bottom: 20px; /* Adjust the margin as needed */
        }
        </style>
        """

        # Inject the custom CSS style
        st.markdown(custom_css, unsafe_allow_html=True)
        
        # Custom CSS
        custom_css = """
        <style>
        
        .custom-info {
            font-size: 22px; /* Adjust the size as needed */
            background-color: #D3DFFF;
            color: black; /* Adjust the color as needed */
            margin-bottom: 20px; /* Adjust the margin as needed */
            border-radius: 10px; /* Adjust border radius as needed */
            padding: 30px; /* Adjust padding as needed */
        }
        </style>
        """

        # Inject the custom CSS style
        st.markdown(custom_css, unsafe_allow_html=True)
        

        # Title
        st.markdown('<h1 class="custom-about">About this App</h1>', unsafe_allow_html=True)
        
        # Info
        st.markdown('<p class="custom-info">The purpose of this app is to make new predictions for an individual\'s fall risk based on six variables. In order to make predictions fill in the variables in the side bar and click the run prediction button. Prediction results will appear at the bottom of the app after calculating.</p>', unsafe_allow_html=True)
        
        with st.container(border=False):
            col1, col2 = st.columns([0.7,0.3])
            
            with col1:
            
                # Title
                st.markdown('<h1 class="custom-about">Definition of a Fall</h1>', unsafe_allow_html=True)
                
                # Info
                st.markdown('<p class="custom-info">A fall is defined as a “sudden, not intentional, and unexpected movement from orthostatic position, from seat to position, or from clinical position”.</p>', unsafe_allow_html=True)

                # Title
                st.markdown('<h1 class="custom-about">Importance of Falls in Hospitals</h1>', unsafe_allow_html=True)
                
                # Info
                st.markdown('<p class="custom-info">Falls directly cost hospitals $50 billion annually in the U.S. Patients who fall while in the hospital have $13,316 higher operational costs. Costs associated with falls are not re-imbursed by Medicare or Medicaid.</p>', unsafe_allow_html=True)

            with col2:
                # Title
                st.markdown('<h1 class="custom-about">Fall Risk Factors</h1>', unsafe_allow_html=True)
                
                # Info
                st.markdown('<p class="custom-info">1. Older than 85<br>2. Weight<br>3. History of falls<br>4. Mobility problems<br>5. Use of assistive devices<br>6. Medications<br>7. Mental status<br>8. Incontinence<br>9. Vision impairment</p>', unsafe_allow_html=True)



with col3:
    # Path to the local GIF file
    gif_path = 'Visuals/fall.gif'
    # Display the GIF
    st.image(gif_path, use_column_width=True)
st.divider()

                                                        ### STATIC TILES ###

col1, col2, col3 = st.columns(3)

## Calculate value count percents for 'Decision' column
# Calculate the count of each unique value
value_counts = data['decision'].value_counts()

# Calculate the percentage of each value
percentages = (value_counts / len(data['decision']) * 100).round(2)


with col1:
        
    ## Updating count for total admitted
    total_row = data.shape[0]

    # Custom CSS
    custom_css = """
    <style>
    .custom-container {
        background-color: #8EAEFF; /* Change to desired background color */
        padding: 30px; /* Adjust padding as needed */
        border-radius: 15px; /* Adjust border radius as needed */
        border: 1px solid #ccc; /* Adjust border as needed */
        }

    .custom-admitted {
        font-size: 70px; /* Adjust the size as needed */
        font-weight: bold;
        color: #333; /* Adjust the color as needed */
        margin-bottom: 40px; /* Adjust the margin as needed */
        text-align: center; /* Center the text */
    }
    </style>
    """

    # Inject the custom CSS style
    st.markdown(custom_css, unsafe_allow_html=True)

    ## Text in tile
    container_html = '''
    <div class="custom-container">
        <h1 class="custom-admitted">Total Patients Admitted</h1>
        <p class="custom-admitted">{total_row}</p>
    </div>
    '''.format(total_row=data.shape[0])

    # Inject the custom container HTML
    st.markdown(container_html, unsafe_allow_html=True)
    
with col2:

    # Custom CSS
    custom_css = """
    <style>
    .custom-fall {
        font-size: 70px; /* Adjust the size as needed */
        font-weight: bold;
        color: #333; /* Adjust the color as needed */
        margin-bottom: 50px; /* Adjust the margin as needed */
        text-align: center; /* Center the text */
    }
    </style>
    """

    # Inject the custom CSS style
    st.markdown(custom_css, unsafe_allow_html=True)

    ## Text in tile
    container_html = '''
    <div class="custom-container">
        <h1 class="custom-fall">Fall Rate</h1>
        <p class="custom-fall">{percent}%</p>
    </div>
    '''.format(percent=percentages[0])

    # Inject the custom container HTML
    st.markdown(container_html, unsafe_allow_html=True)

with col3:
        
    # Custom CSS
    custom_css = """
    <style>
    .custom-stumble {
        font-size: 70px; /* Adjust the size as needed */
        font-weight: bold;
        color: #333; /* Adjust the color as needed */
        margin-bottom: 50px; /* Adjust the margin as needed */
        text-align: center; /* Center the text */
    }
    </style>
    """

    # Inject the custom CSS style
    st.markdown(custom_css, unsafe_allow_html=True)

    # Text in tile
    container_html = '''
    <div class="custom-container">
        <h1 class="custom-fall">Stumble Rate</h1>
        <p class="custom-fall">{percent}%</p>
    </div>
    '''.format(percent=percentages[1])

    # Inject the custom container HTML
    st.markdown(container_html, unsafe_allow_html=True)


        
                                                 ### EDA HISTOGRAM ###
## Reformat columns in dataframe
data.rename(columns={'distance':'Distance (cm)','hrv':'Heart Rate (bpm)',
                   'blood_sugar':'Blood Sugar Level (mg/dL)',
                   'spo2':'SpO2 (%)', 'accelerometer':'Accelerometer',
                   'pressure':'Pressure','decision':'Decision'}, inplace=True)

# Custom CSS
custom_css = """
<style>
.custom-heading {
    font-size: 60px; /* Adjust the size as needed */
    font-weight: bold;
    color: #333; /* Adjust the color as needed */
    margin-bottom: 20px; /* Adjust the margin as needed */
}
</style>
"""

# Inject the custom CSS style
st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="custom-heading">Data Exploration</h1>', unsafe_allow_html=True)
   
col1, col2, col3 = st.columns([0.3,0.2,0.4])
with col1:
        
    ## Title for visual
    # Custom CSS
    custom_css = """
    <style>
    .custom-subheading {
        font-size: 30px; /* Adjust the size as needed */
        color: #333; /* Adjust the color as needed */
        font-weight: normal;
        margin-bottom: 5px; /* Adjust the margin as needed */
        text-align: center;
    }
    </style>
    """

    # Inject the custom CSS style
    st.markdown(custom_css, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="custom-subheading">Distribution of Continuous Variables</h1>', unsafe_allow_html=True)

    # Melt the DataFrame
    long_df = pd.melt(data, value_vars=['Distance (cm)', 'Heart Rate (bpm)',
                     'Blood Sugar Level (mg/dL)','SpO2 (%)'],
                      var_name='Variable', value_name='Value')

    # Create the histogram
    fig = px.histogram(long_df, x='Value', color='Variable', barmode='overlay')

    # Update the layout
    fig.update_layout( 
        xaxis_title=dict(text='Value', font=dict(size=25)),
        yaxis_title=dict(text='Count', font=dict(size=25)),
        xaxis = dict(tickfont = dict(size=16)),
        yaxis = dict(tickfont = dict(size=16)),
        legend_title_text='Variables',
        width=800,
        height=800)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

                                        ### EDA COUNT PLOT ###
with col2:
        
    ## Title for plot
    st.markdown('<h1 class="custom-subheading">Categorical Varibales Grouped By Fall Status</h1>', unsafe_allow_html=True)

    # Mapping from numeric to descriptive categories
    category_mapping = {
        0: 'No Fall',
        1: 'Stumbled',
        2: 'Fall'
    }


    # Reshape the DataFrame
    df_melted = data.melt(id_vars=['Decision'], value_vars=['Pressure', 'Accelerometer'], 
                var_name='Type')
    df_melted = df_melted.drop(columns='value')
    df_melted = df_melted.replace(category_mapping)

    # Create the count plot
    fig = px.histogram(df_melted, x='Decision', color='Type', pattern_shape='Decision', 
                       labels={'Type': 'Type', 'Value': 'Value','count': 'Count'})

    # Update the layout
    fig.update_layout(
        xaxis_title=dict(text='Fall Status', font=dict(size=25)),
        yaxis_title=dict(text='Count', font=dict(size=25)),
        xaxis = dict(tickfont = dict(size=16)),
        yaxis = dict(tickfont = dict(size=16)),
        legend_title_text='Variables',
        width=800,
        height=775)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)  

with col3:
    # Compute the correlation matrix
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.where(~mask, np.nan)
    custom_color_scale = [
        [0.0, '#216CD3'],  # Start at 0
        [0.5, 'white'],  # Transition through white at 0.5
        [1.0, '#FF7B13']   # End at blue at 1
    ]
    
    ## Title for plot
    st.markdown('<h1 class="custom-subheading">Correlations Between Variables</h1>', unsafe_allow_html=True)

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
        title_font={'size': 25},  # Font size for x-axis title
        tickfont={'size': 16}  # Font size for x-axis tick labels
    ),
    yaxis=dict(
        tickmode='linear',
        title='Features',  # y-axis title
        title_font={'size': 25},  # Font size for y-axis title
        tickfont={'size': 16}  # Font size for y-axis tick labels
    ),
    width=1000,
    height=800
    )

    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))


    # Display the heatmap in Streamlit
    st.plotly_chart(fig, use_container_width=True)

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

prediction = model.predict(user_input)

prediction_button = st.sidebar.button('**Run Prediction**')

if prediction_button:
    on_click=button_clicked()
    
                                                                ### MODEL MATRIX ###
# Title
st.markdown('<h1 class="custom-heading">Machine Learning</h1>', unsafe_allow_html=True)

# Get predictions
pred = model.predict(X_test)

target_names = ['No Fall','Stumble Risk','Fall Risk']

# Display classification report
## Title for plot
st.markdown('<h1 class="custom-subheading">Overall Model Performance</h1>', unsafe_allow_html=True)
report = pd.DataFrame(classification_report(y_test, pred, target_names=target_names, output_dict=True)).transpose()
st.dataframe(report, use_container_width=True)

col1, col2 = st.columns(2)

## Calculating SHAP values
explainer = shap.TreeExplainer(model[-1], X_train)
shap_values = explainer(user_input)


with col1:
    
    ## Title for plot
    st.markdown('<h1 class="custom-subheading">Average Feature Importances</h1>', unsafe_allow_html=True)
    
    feature_names = model[:-1].get_feature_names_out()
    feature_importance = pd.Series(model[-1].feature_importances_, index=feature_names)

    # Create a DataFrame from the series for easier handling with Plotly
    feature_importance_df = pd.DataFrame({'Feature': feature_importance.index, 'Importance': feature_importance.values})

    # Sort the DataFrame by importance
    sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    # Visualize using Plotly Express - showing all features
    fig = px.bar(sorted_feature_importance_df, x='Importance', y='Feature', orientation='h')
    fig.update_layout(
        xaxis=dict(
        title='Feature Importance Value',  # x-axis title
        title_font={'size': 25},  # Font size for x-axis title
        tickfont={'size': 16}  # Font size for x-axis tick labels
    ),
    yaxis=dict(
        title='Feature Name',  # y-axis title
        title_font={'size': 25},  # Font size for y-axis title
        tickfont={'size': 16}  # Font size for y-axis tick labels
    ),
    width=800,
    height=525
    )

    # Show the plot
    st.plotly_chart(fig)
    
with col2:    
    
    ## Title for plot
    st.markdown('<h1 class="custom-subheading">Feature Importances for Current Prediction </h1>', unsafe_allow_html=True)
    
    fig = shap.plots.waterfall(shap_values[0, :, 0])
    st.pyplot(fig)


                                                ### MACHINE LEARNING TILES ###
col1, col2, col3 = st.columns(3)

with col1:
    
    st.markdown("""
        <style>
        .container {
            padding: 20px;
            border-radius: 15px;
            height: 350px;  /* Adjust the height as needed */
            margin: 10px 0;
        }
        .no-risk {
            background-color: #d4edda;  /* Light green background */
            color: #155724;  /* Dark green text */
            text-align: center; /* Center the text */
        }
        .orange-text {
            background-color: #fff3cd;  /* Light orange background */
            color: orange;
            text-align: center; /* Center the text */
        }
        .red-text {
            background-color: #f8d7da;  /* Light red background */
            color: red;
            text-align: center; /* Center the text */
            text-transform: uppercase;
        }
        h2 {
         font-size: 70px; /* Adjust the size as needed */
         font-weight: bold;
         color: #333; /* Adjust the color as needed */
         margin-bottom: 20px; /* Adjust the margin as needed */
         text-align: center; /* Center the text */
         margin-top: 0;
        }
       .result-text {
        font-size: 70px;  /* Adjust the font size as needed */
        }
        </style>
        """, unsafe_allow_html=True)

    with st.spinner('Calculating Prediction...'):
        time.sleep(2)
        
    result = interpret_prediction(prediction)
    
    # Display markdown
    st.markdown(result, unsafe_allow_html=True)
           

with col2:

#         st.subheader('Prediction Probability')
        # Formatting the probability output for clarity
        prediction_probability = model.predict_proba(user_input)

        # Extract the highest probability and its corresponding class
        predicted_class_index = prediction_probability.argmax()
        predicted_class_probability = prediction_probability.max()

        # Define the classes (assumes model.classes_ contains class labels)
        predicted_class = model.classes_[predicted_class_index]
        
#         # Show the predicted class and its probability
#         st.write(f"{predicted_class_probability * 100:.2f}%")
        
        # CSS styles
        st.markdown("""
            <style>
            .custom-container {
                padding: 0px;
                border-radius: 15px;
                height: 350px;  /* Adjust the height as needed */
                margin: 10px 0;
            }
            .custom-heading {
                font-size: 70px;
                font-weight: bold;
                text-align: center; /* Center the text */
                margin-bottom: 10px;
            }
            .custom-content {
                font-size: 70px;
                text-align: center; /* Center the text */
            }
            </style>
            """, unsafe_allow_html=True)

        # Custom container using st.markdown with HTML and CSS
        st.markdown(f"""
            <div class="custom-container">
                <div class="custom-heading">Prediction Probability</div>
                <div class="custom-content">
                    {predicted_class_probability * 100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
with col3:
        
       # Assuming you want to analyze the first (and only) instance in the input
        instance_index = 0

        # Extract the SHAP values for the predicted class and instance
        predicted_class_shap_values = shap_values.values[instance_index, :, predicted_class_index]
        
        # Find the most important feature for the predicted class
        most_important_feature_index = np.argmax(np.abs(predicted_class_shap_values))
        most_important_feature = user_input.columns[most_important_feature_index]
        
        
        # CSS styles
        st.markdown("""
            <style>
            .custom-container {
                padding: 0px;
                border-radius: 15px;
                height: 350px;  /* Adjust the height as needed */
                margin: 10px 0;
            }
            .custom-heading {
                font-size: 70px;
                font-weight: bold;
                text-align: center; /* Center the text */
                margin-bottom: 10px;
            }
            .custom-content {
                font-size: 70px;
                text-align: center; /* Center the text */
            }
            </style>
            """, unsafe_allow_html=True)

        # Custom container using st.markdown with HTML and CSS
        st.markdown(f"""
            <div class="custom-container">
                <div class="custom-heading">Most Important Feature</div>
                <div class="custom-content">
                    {most_important_feature}
                </div>
            </div>
            """, unsafe_allow_html=True)
