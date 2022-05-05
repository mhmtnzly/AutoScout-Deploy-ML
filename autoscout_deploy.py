#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May 07 12:00:00 2022
@author: PyBoys
"""

#Import libraries


import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
current_year = date.today().year

#load the model from disk
import joblib
filename = 'finalized_model.sav'
model = joblib.load(filename)

sc = joblib.load(open('scaler.sav', 'rb'))

#Import python scripts
from preprocessing import preprocess,make_model_dict, n_dict

def main():
    #Setting Application title
    st.title('PyBoys AutoScout Model App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict AutoScout use case.
    The application is functional for online prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Default"))
    st.sidebar.info('This app is created to predict AutoScout use case')
    


    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection
        st.subheader("The Most Important Features:")
        make_model_label = st.selectbox('Make and Model of the Car' , (make_model_dict))
        first_registration = st.number_input('First Registration Year of the Car', min_value=1940, max_value=2021, value=2000)
        empty_weight = st.number_input('Empty Weight of the Car', min_value=0, value =0) #### ok
        mileage = st.number_input('Mileage of the Car',value = 0)
        Power = st.number_input('Power of the Car',value = 0)
        fuel_consumption = st.number_input('Fuel Consumption of the Car',value = 0.0)
        co2_emissions = st.number_input('Co2 emission per km', value = 0)
        engine_size = st.number_input('Engine Size of the Car',value=0) 
        gears = st.selectbox('Gear of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        seats = st.selectbox('Seat of the Car', ('1','2','3', '4','5','6','7', '8','9')) 
        doors = st.selectbox('Door of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        cylinder = st.selectbox('Cylinder of the Car', ('1','2','3', '4','5','6','7', '8','9','10'))    
        warranty_months = st.number_input('How many months does it have ?', value = 0)

        age = current_year - first_registration 
        total_consumption =  fuel_consumption * mileage/100 
        total_co2_emissions = mileage * co2_emissions

        data = {
                'make_model_label': make_model_dict[make_model_label],
                'mileage': mileage, 
                'seats': n_dict[seats],
                'doors': n_dict[doors],
                'warranty_months': warranty_months,
                'Power': Power,
                'engine_size': engine_size,
                'Gears': n_dict[gears],
                'Cylinders': n_dict[cylinder],
                'Empty_weight':empty_weight,
                'fuel_consumption': fuel_consumption,
                'co2_emissions' : co2_emissions,
                'first_registration_years': age,# 1
                'mileage_years': mileage/age,
                'total_consumption': total_consumption, #
                'total_co2_emissions': total_co2_emissions
                }
      #           'price', 'mileage', 'seats', 'doors', 'warranty_months', 'Power',
      #  'engine_size', 'Gears', 'Cylinders', 'Empty_weight', 'fuel_consumption',
      #  'co2_emissions', 'first_registration_years', 'make_model_label',
      #  'mileage_years', 'total_consumption', 'total_co2_emissions'


        
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs
        #preprocess_df = preprocess(features_df, 'Online')


        features_df_scaled = sc.transform(features_df)

        prediction = np.exp(model.predict(features_df_scaled))

        if st.button('Predict'):
            st.warning(prediction)
        

if __name__ == '__main__':
        main()