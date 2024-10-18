
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
lin_reg = joblib.load("50_startup_lin_reg.pkl")

# Load the pipeline
full_pipeline = joblib.load("50_startup_full_pipeline.pkl")

# Load the data
df = pd.read_csv("50_Startups.csv")

# Create a title and sub-title
st.title(" Welcome to my 1st ML App ")

st.write("""
This app predict profit value of a certain startup project within 3 USA states**!
""")

# Take the input from the user
RD_Spend = st.number_input("R&D Spend", min_value= float(df['R&D Spend'].min()), max_value=float(df['R&D Spend'].max()), value= 165349.2)
Administration = st.number_input("Administration Spend", min_value= float(df['Administration'].min()), max_value=float(df['Administration'].max()), value= 136897.8)
Marketing_Spend = st.number_input("Marketing Spend", min_value= float(df['Marketing Spend'].min()), max_value=float(df['Marketing Spend'].max()), value= 471784.1)
State = st.selectbox('Select your State', ('New York', 'California', 'Florida'))

# Store a dictionary into a variable
user_data = {'R&D Spend': RD_Spend,
'Administration': Administration,
'Marketing Spend': Marketing_Spend,
'State': State}


# Transform the data into a data frame
features = pd.DataFrame(user_data, index=[0])

# Pipeline
features_prepared = full_pipeline.transform(features)

# Predict the output
prediction = lin_reg.predict(features_prepared)[0]

# Set a subheader and display the prediction
st.subheader('Profit Prediction')
st.markdown('''# $ {} '''.format(round(prediction), 2))
