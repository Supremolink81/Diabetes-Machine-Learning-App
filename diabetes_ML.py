# This is a stock market web app that uses Python to
# display information on stocks through graphs and data

import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

#Create title and sub-title
st.write("""
# Diabetes Detection

Detect if someone has diabetes using machine learning and python!
""")

#Create and open an image
image = Image.open('C:/Users/arioz/OneDrive/Documents/Programming/Python Projects/Diabetes Machine Learning App/machine_learning_diabetes_image.png')
st.image(image, caption = "ML", use_column_width = True)

#Get data
data = pd.read_csv("C:/Users/arioz/OneDrive/Documents/Programming/Python Projects/Diabetes Machine Learning App/diabetes.csv")
st.subheader("Data Information")

#Show data as table
st.dataframe(data)

#Show statistics on data
st.write(data.describe())

#Show data as chart
chart = st.bar_chart(data)

#Split data into independent x and dependent y variables
X = data.iloc[:,0:8].values
Y = data.iloc[:,-1].values

#Split dataset into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Retrieve user input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinThickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('Body Mass Index (BMI)', 0.0, 67.1, 32.0)
    diabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function (DPF)', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    #Store a dictionary into a variable
    user_data = {
        "pregnancies" : pregnancies,
        "glucose" : glucose,
        "blood_pressure" : bloodPressure,
        "skin_thickness" : skinThickness,
        "insulin" : insulin,
        "BMI" : BMI,
        "DPF" : diabetesPedigreeFunction,
        "Age" : age
                 }
    
    #Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

#Store user input in a variable
user_input = get_user_input()

#Set a subheader and display user input
st.subheader("User Input:")
st.write(user_input)

#Create and train the model
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, Y_train)

#Show the model's metrics
st.subheader("Model Test Accuracy Score:")
st.write(str(accuracy_score(Y_test, randomForestClassifier.predict(X_test)) * 100)+"%")

#Store the model's predictions in a variable
prediction = randomForestClassifier.predict(user_input)

#Set a subheader and display the classification
st.subheader("Classification:")
st.write(prediction)