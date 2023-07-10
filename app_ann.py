import pandas as pd
import streamlit as st
import keras
import numpy as np
#from tensorflow import keras
from PIL import Image
from keras.models import Sequential
#from keras_preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

     ##    python -m streamlit run app_ann.py
    ##    cd C:\Users\yener\OneDrive\Desktop\heart_failure_clinical_records_dataset

##loading the ann model
model = keras.models.load_model("ann_model")

## load the copy of the dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

## set page configuration
st.set_page_config(page_title = 'Heart Failure Decease Prediction')

## add page title and content
st.title('Heart Failure Decease Prediction ')
st.write('Please Enter Set of Information Requested Below:')

## add image
image = Image.open("1_6nE1MADqsFutGS-MZeYXig.jpeg")
st.image(image, use_column_width = True)

#user input 
age = st.number_input('Enter Age: ', min_value=40, max_value=100)
anaemia = st.number_input('Enter Anaemia, 1 for positive, 0 for negative: ', min_value=0, max_value=1)
creatinine_phosphokinase = st.number_input('Enter Creatinine level, between 23 and 7861: ',min_value=23, max_value=7861)
diabetes = st.number_input('Enter Diabetes, 1 for positive, 0 for negative: ',min_value=0, max_value=1)
ejection_fraction = st.number_input('Enter Ejection Fraction, between 14 and 80: ',min_value=14, max_value=80)
high_blood_pressure = st.number_input('Enter High Blood Pressure, 1 for positive, 0 for negative: ',min_value=0, max_value=1)
platelets = st.number_input('Enter Platelets level, between 25100 and 85000: ',min_value=25100, max_value=850000)
serum_creatinine = st.number_input('Enter Serun Creatinine level, between 0.5 and 9.4: ',min_value=0.5, max_value=9.4)
serum_sodium = st.number_input('Enter Serun Sodium level, between 113 and 148: ',min_value=113, max_value=148)
sex = st.number_input('Enter Gender, 0 for Female; 1 for Male: ',min_value=0, max_value=1)
smoking = st.number_input('Enter Smoking, 0 for non-smoker; 1 for smoker: ',min_value=0, max_value=1)
time = st.number_input('Enter follow up period for days between 4 and 285 :',min_value=4, max_value=285)

a = np.array([[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])

## convert text to numerical values
##word_index = {word: index for index, word in enumerate(df.columns[:-1])}
##numerical_values = [a]

## pad the numerical values so that it can have a unique shape as the training data 
##padded_value = pad_sequences([numerical_values],maxlen=3000)

## make the prediction
if st.button('Predict'):
    prediction = model.predict(a)

    if prediction > 0.5:
        st.write('The follow up prediod resulted with decease')
    else: 
        st.write('The follow up prediod not resulted with decease')
