import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("💼 Customer Churn Prediction App")
st.markdown("Enter customer details and get the predicted churn probability.")

# ------------------ Load Model & Preprocessors ------------------
model = tf.keras.models.load_model("model.h5", compile=False)  # compile=False avoids HDF5 errors

with open("onehot_encoder_geo.pk1", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("label_encoder_gender.pk1", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("scaler.pk1", "rb") as f:
    scaler = pickle.load(f)

# ------------------ User Inputs ------------------
st.sidebar.header("Customer Details")
geography = st.sidebar.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)

# Replace number_input with text_input to hide +/- buttons
age = st.sidebar.text_input("Age", value="30")
age = int(age) if age.isdigit() else 30

balance = st.sidebar.text_input("Balance", value="50000")
balance = int(balance) if balance.isdigit() else 50000

credit_score = st.sidebar.text_input("Credit Score", value="600")
credit_score = int(credit_score) if credit_score.isdigit() else 600

estimated_salary = st.sidebar.text_input("Estimated Salary", value="50000")
estimated_salary = int(estimated_salary) if estimated_salary.isdigit() else 50000

tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1])

# ------------------ Prepare Input Data ------------------
def prepare_input():
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    final_input_df = pd.concat([input_df, geo_encoded_df], axis=1)
    return final_input_df

# ------------------ Predict Function ------------------
def predict_churn(input_df):
    input_scaled = scaler.transform(input_df)
    prediction_prob = model.predict(input_scaled)[0][0].item()
    return prediction_prob

# ------------------ Predict Button ------------------
final_input_df = prepare_input()
if st.button("Predict Churn"):
    prediction_prob = predict_churn(final_input_df)

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{prediction_prob:.2f}**")

    if prediction_prob > 0.5:
        st.error("❌ The Customer is likely to Churn.")
    else:
        st.success("✅ The Customer is not likely to Churn.")

# ------------------ Show Input Data ------------------
st.subheader("Customer Input Data")
st.table(final_input_df)
