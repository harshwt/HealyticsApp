import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

newxmodel = joblib.load("./diabetes_model.pkl")
scaling = joblib.load("./scaler.pkl")

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

age_bmi = age * bmi
glucose_bmi = glucose / (bmi if bmi != 0 else 1)
age_bp = age * bp
age_skin = age * skin
age_insulin = age * insulin
age_preg = age * preg
age_dpf = age * dpf

ip = np.array([[preg, glucose, bp, age, skin, insulin, bmi, dpf,
                age_bmi, glucose_bmi, age_bp, age_skin, age_insulin, age_preg, age_dpf]])
scaled_ip = scaling.transform(ip)

  
if st.sidebar.button('Predict'):
  prediction = newxmodel.predict(scaled_ip)
  probability = newxmodel.predict_proba(scaled_ip)
  prob = probability[0][1]
  prob_text = f"Likelihood of Diabetes: {prob*100:.2f}%"
  st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
  st.success(prob_text)

  if prob < 0.3:
    risk_level = "Low Risk"
    color = "green"
    st.markdown(f"<p style='color:green; font-size:20px;'> Low Risk : <b>{prob_text}</b></p>", unsafe_allow_html=True)
    st.markdown("<h3>Recommendations:</h3>", unsafe_allow_html=True)
    st.write("- Keep maintaining a healthy lifestyle.")
    st.write("- Regular checkups once or twice a year.")
    st.write("- Exercise and balanced diet help prevent future risk.")
    
  elif prob < 0.6:
    st.markdown(f"<p style='color:orange; font-size:20px;'> Moderate Risk : <b>{prob_text}</b></p>", unsafe_allow_html=True)
    risk_level = "Moderate Risk"
    color = "orange"
    st.markdown("<h3>Recommendations:</h3>", unsafe_allow_html=True)
    st.write("- Monitor blood glucose regularly.")
    st.write("- Maintain a balanced diet and exercise routinely.")
    st.write("- Consult a doctor if risk factors increase.")
  else:
    st.markdown(f"<p style='color:red; font-size:20px;'>Person is Prone to diabetes: <b>{prob_text}</b></p>", unsafe_allow_html=True)
    risk_level = "High Risk"
    color = "red"
    st.markdown("<h3>Recommendations:</h3>", unsafe_allow_html=True)
    st.write("- Consult a doctor for a proper checkup immediately.")
    st.write("- Maintain a healthy diet (low sugar, balanced meals).")
    st.write("- Exercise regularly (30 mins/day).")
    st.write("- Monitor blood glucose levels frequently.")
    
  '''
  '''
  if prob >:
    st.markdown(f"<p style='color:red; font-size:20px;'>Person is Prone to diabetes: <b>{prob_text}</b></p>", unsafe_allow_html=True)
    st.markdown("<h3>Recommendations:</h3>", unsafe_allow_html=True)
    st.write("- Consult a doctor for a proper checkup.")
    st.write("- Maintain a healthy diet (low sugar, balanced meals).")
    st.write("- Exercise regularly (30 mins/day).")
    st.write("- Monitor blood glucose levels frequently.")
    
  else:
    st.markdown("<h3>Recommendations:</h3>", unsafe_allow_html=True)
    st.write("- Keep maintaining a healthy lifestyle.")
    st.write("- Regular checkups once or twice a year.")
    st.write("- Exercise and balanced diet help prevent future risk.")
  '''

st.write("This app uses a machine learning model trained on the PIMA Indian Diabetes dataset.")
st.write("Continuous health tracking: Users can monitor their health regularly by integrating fitness trackers, diet plans, and lifestyle monitoring apps, enabling proactive diabetes management.")
st.write("It can help rural areas without immediate access to doctors and urban users to track diabetes risk early, raising awareness as India faces increasing diabetes cases.")

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'Age', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
normal_ranges = [0, 120, 80, 20, 100, 25, 20, 40] 
user_values = [preg, glucose, bp, age, skin, insulin, bmi, dpf]

x = np.arange(len(features))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - 0.2, user_values, width=0.4, label='Your Input', color='orange')
ax.bar(x + 0.2, normal_ranges, width=0.4, label='Normal Range', color='lightgreen')

ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right')
ax.set_ylabel("Values")
ax.set_title("Your Input vs Healthy Ranges")
ax.legend()
st.pyplot(fig)


matrix = np.array([[45, 5],
                        [7, 23]])

st.subheader('Model Performance: ')
st.write("Confusion Matrix:")
#getting matrix report
fig, ax = plt.subplots()
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes','Diabetes'],
            yticklabels=['No Diabetes','Diabetes'],
            ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)
