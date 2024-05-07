from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
tf.random.set_seed(42)

st.header( ':🌾:',divider='rainbow')
st.header( ':blue[Crop-recommendation] ')

model = tf.keras.models.load_model('crop.h5')

N = st.number_input("Insert N")
st.write("The current number is ", N)

P = st.number_input("Insert P",key='p')
st.write("The current number is ", P)

K = st.number_input("Insert K",key='K')
st.write("The current number is ", K)

temperature = st.number_input("Insert Temp",key='temp')
st.write("The current number is ", temperature)

humidity = st.number_input("Insert Humidity",key='hum')
st.write("The current number is ", humidity)

ph = st.number_input("Insert ph",key='ph')
st.write("The current number is ", ph)

rainfall = st.number_input("Insert rainfall",key='rain')
st.write("The current number is ", rainfall)





new_data = pd.DataFrame({
    'N': [N],  # Example values, replace with your actual data
    'P': [P],
    'K': [K],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})
 
   
b1=st.button('ShowPrediction',key=1)
if b1:
        scaler = StandardScaler()
        new_data_scaled = scaler.fit_transform(new_data)


        predictions = model.predict(new_data_scaled)


        label_encoder = LabelEncoder()

        labels =['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
            'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
        label_encoder.fit(labels)


        predicted_labels_encoded = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

        print("Predicted labels (encoded):")
        print(predicted_labels_encoded)

        print("Predicted labels (decoded):")
        print(predicted_labels)
        st.title(predicted_labels)