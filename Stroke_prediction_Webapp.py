# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:38:51 2023

@author: johnb
"""
import numpy as np
import streamlit as st
import joblib

st.set_option('deprecation.showfileUploaderEncoding',False) 

# loading the saved model

loaded_model = joblib.load(open('trained_model.pkl', 'rb'))



# creating a function for Prediction

def stroke_prediction(input_data):
    
  input_data = (1,56.012907,0,0,0,1,3.695411,-0.059290,0,0,1,0,0)

# changing the input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  prediction = loaded_model.predict(input_data_reshaped)
  print(prediction)

  if (prediction[0] == 0):
   return 'The person does not have the tendency of having a stroke'
  else:
   return 'The person has a tendency of having a stroke'
  
    
  
def main():
    # giving a title
    st.title('Stroke Prediction Web App')

    # getting the input data from the user
    Hypertension = st.text_input('Hypertension value')
    Glucose_level = st.text_input('Glucose Level')
    Gender = st.text_input('Gender value')   
    BMI = st.text_input('BMI value')
    Ever_married = st.text_input('Ever married value')
    Work_type = st.text_input('Work type value')
    Age = st.text_input('Age of the Person')
    Residence_type = st.text_input('Residence type')
    Heart_disease= st.text_input('Do you or have you had a heart disease')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('stroke Test Result'):
        diagnosis = stroke_prediction([Heart_disease, Hypertension, Glucose_level, Gender, BMI, Ever_married, Work_type, Residence_type, Age])
       
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    