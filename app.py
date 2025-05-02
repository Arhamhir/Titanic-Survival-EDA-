import joblib 
import streamlit as st
import numpy as np

model = joblib.load('mymodel.pkl')
scaler = joblib.load('myscaler.pkl')

st.header('Titanic Survival Prediction')

fare = st.number_input('Enter fare', min_value=10.0, max_value=550.0, value=50.0)
gender = st.radio('Gender', ['M', 'F'])
status = st.radio('Travelling status', ['With Someone', 'Alone'])
class_type = st.radio('Class', ['First', 'Second', 'Third'])

adult_male = 1 if gender == 'M' else 0
alone = 1 if status == 'Alone' else 0

class_First = 1 if class_type == 'First' else 0
class_Second = 1 if class_type == 'Second' else 0
class_Third = 1 if class_type == 'Third' else 0

if st.button('Predict'):
    user_input = [fare, adult_male, alone, class_First, class_Second, class_Third]
    user_input = np.array([user_input])

    scaled_input = scaler.transform(user_input) 

    prediction = model.predict(scaled_input)
    st.success(f'Survival Prediction: {"Survived" if prediction[0] == 1 else "Did not survive"}')
