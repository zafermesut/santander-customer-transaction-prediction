import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import xgboost 

st.title("Santander Customer Transaction Prediction")
st.write("""
    This app predicts whether customers will take a certain action in the future based on the CSV file you upload.  
    Please upload a CSV file in the appropriate format.
""")

model = joblib.load("xgb_model.joblib")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

button = st.button("Predict with Ready Rest Data")   

if button:
    data = pd.read_csv('test_sample.csv')
    
    st.write("Preview of loaded data:")
    st.write(data.head())  
    
    predictions = model.predict(data)
    
    data['Prediction'] = predictions
    
    st.write("Prediction Results:")
    st.write(data[['Prediction']])  
    
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(data)

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Preview of loaded data:")
    st.write(data.head())  
    
    if st.button("Predict"):
        predictions = model.predict(data)
        
        data['Prediction'] = predictions
        
        st.write("Prediction Results:")
        st.write(data[['Prediction']])  
        
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
