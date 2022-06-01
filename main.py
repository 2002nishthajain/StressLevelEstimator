import streamlit as st
import pickle
import numpy as np
def load_model():
    with open('xgb_model.pkl','rb') as file :
        mod = pickle.load(file)
    return mod

model_dt = load_model()
model = model_dt['model']

def predict_stresslevel():
    st.title("Stress Level Predictor")
    st.write("Give Input to following parameter")
    sr = st.slider("Enter Snoring rate",45,100,55)
    rr = st.slider("Enter Respiration Rate",16,30,20)
    t = st.slider("Enter Temprature",85,105,96)
    lm = st.slider("Enter Limb Movement",4,19,10)
    bo = st.slider("Enter Blood oxygen level",40,99,90)
    sh = st.slider("Enter Sleeping hours",0,10,6)
    hr = st.slider("Enter Heart Rate",50,100,72)
    X = [[sr,rr,t,lm,bo,sh,hr]]
    predict_btn = st.button("Predict the Stress Level")
    if predict_btn:
        result = model.predict(np.array(X))
        st.subheader(f"The predicted Stress Level is: {result[0]}")

