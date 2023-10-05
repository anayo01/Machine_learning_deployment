# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:14:06 2023

@author: johnb
"""

import numpy as np
import pickle 


# loading the saved model
loaded_model = pickle.load(open('C:/Users/johnb/OneDrive/Desktop/StrokeML_Model/trained_model.sav', 'rb'))



##Prediction with a slice of the dataset to see how accurate it can predict values from the trained dataframe
##Adaboost

input_data = (1,56.012907,0,0,0,1,3.695411,-0.059290,0,0,1,0,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person does not have the tendency of having a stroke')
else:
  print('The person has a tendency of having a stroke')