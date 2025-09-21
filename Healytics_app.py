import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")
st.write("Enter the following details to predict diabetes:")
st.sidebar.header("Input Parameters")

preg = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=0, step=1)
bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=0, step=1)
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=1)
skin = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=800, value=0)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=60.0, value=0.0, step=0.10)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

# Correct the input array creation and scaling
ip = np.array([[preg, glucose, bp, age, skin, insulin, bmi, dpf]])
scaled_ip = scaling.transform(ip)

if st.sidebar.button('Predict'):
  prediction = newxmodel.predict(scaled_ip)
  if prediction[0] == 1:
    st.write("The person is prone to diabetes!")
  else:
    st.write("The person is not at risk")

st.subheader('Performance and Scale: ')
y_pred = newxmodel.predict(xtest)
st.write(confusion_matrix(ytest, y_pred))
st.write(classification_report(ytest, y_pred))

#getting matrix report
cm = confusion_matrix(ytest, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes','Diabetes'],
            yticklabels=['No Diabetes','Diabetes'],
            ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)
