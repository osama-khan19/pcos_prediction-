import streamlit as st
import pandas as pd
import pickle


with st.sidebar:
    st.header(" Input Guide")
    st.markdown("""
    - **Age**: Patient's age in years (12–50).  
    - **BMI**: Body Mass Index, normal range 18.5–24.9.  
    - **Menstrual Irregularity**: 0 = No, 1 = Yes.  
    - **Testosterone Level (ng/dL)**: Normal range (15–70 ng/dL).  
    - **AFC**: Antral follicle count.  
        - 25–34 years: 10–13 follicles  
        - 35–40 years: 8–10 follicles  
        - 41+ years: 5–7 follicles  
    """)


with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)


st.title(" PCOS Prediction App")
st.write("This app is for academic/demo purposes only. Do not use for medical decisions.")

name = st.text_input("Enter patient name")
age = st.number_input("Age", min_value=10, max_value=60, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=22.0)
cycles = st.number_input("Menstrual Irregularity (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
testosterone = st.number_input("Testosterone Level (ng/dL)", min_value=0, max_value=90, value=50)
afc = st.number_input("AFC Level", min_value=0.0, max_value=40.0, value=10.0)


input_data = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "Menstrual_Irregularity": cycles,      
    "Antral_Follicle_Count": afc,
    "Testosterone_Level(ng/dL)": testosterone
}])


input_data = input_data[feature_names]


scaled_input = scaler.transform(input_data)


if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error(f"Patient {name if name else 'X'} is likely to have PCOS.")
    else:
        st.success(f"Patient {name if name else 'X'} is unlikely to have PCOS.")
