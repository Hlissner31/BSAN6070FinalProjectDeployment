import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ MUST come before any Streamlit command
st.set_page_config(page_title="Income Prediction App", layout="wide")

# Load components
preprocessor = joblib.load("income_preprocessor.pkl")
model = joblib.load("income_xgb_model.pkl")
average_mae = joblib.load("average_mae.pkl")

# Load Excel data for industry and occupation codes
@st.cache
def load_data():
    file_path = 'Mapping.xlsx'  # Adjust this path to your actual file
    industry_df = pd.read_excel(file_path, sheet_name='IND')
    occupation_df = pd.read_excel(file_path, sheet_name='OCC')
    return industry_df, occupation_df

industry_df, occupation_df = load_data()

st.set_page_config(page_title="Income Prediction App", layout="wide")
st.title("Predicted Personal Income")

# Define input form
with st.form("income_form"):
    st.write("### Enter Person's Information")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 0, 120, 30)
        sex = st.selectbox("Sex", [1, 2])
        statefip = st.number_input("State FIPS", 1, 99, 6)
        marital_status = st.number_input("Marital Status (MARST)", 1, 9, 1)
        nchil = st.number_input("Number of Children", 0, 20, 0)
        uhrswork = st.number_input("Hours Worked per Week", 0, 100, 40)

    with col2:
        classwkr = st.number_input("Class of Worker (CLASSWKR)", 0, 99, 27)
        trantime = st.number_input("Transit Time (minutes)", 0, 999, 30)
        transwork = st.number_input("Mode of Transport to Work (TRANWORK)", 0, 99, 10)
        degfield = st.number_input("Degree Field 1 (Encoded)", 0, 999, 231)
        degfield2 = st.number_input("Degree Field 2 (Encoded)", 0, 999, 0)
        speakeng = st.number_input("English Proficiency (Encoded)", 0, 5, 4)
        educ = st.number_input("Education Level (Encoded)", 0, 100, 70)

    with col3:
        race = st.number_input("Race", 1, 9, 1)
        bpl = st.number_input("Birthplace Code (BPL)", 1, 999, 100)
        ancestr1 = st.number_input("Ancestry Code", 0, 999, 100)
        
        # Replace occsoc and ind with dropdowns based on the Excel file
        industry_options = industry_df['Industry Name'].tolist()
        occupation_options = occupation_df['Occupation Name'].tolist()

        selected_industry = st.selectbox("Select an Industry", industry_options)
        selected_occupation = st.selectbox("Select an Occupation", occupation_options)

        # Get the corresponding codes based on user selection
        ind = industry_df[industry_df['Industry Name'] == selected_industry]['Industry Code'].values[0]
        occsoc = occupation_df[occupation_df['Occupation Name'] == selected_occupation]['Occupation Code'].values[0]

        wkswork1 = st.number_input("Weeks Worked Last Year", 1, 7, 6)

    submitted = st.form_submit_button("Predict Income")

# Predict
if submitted:
    input_dict = {
        "AGE": age,
        "SEX": sex,
        "STATEFIP": statefip,
        "MARST": marital_status,
        "NCHILD": nchil,
        "UHRSWORK": uhrswork,
        "CLASSWKR": classwkr,
        "TRANTIME": trantime,
        "TRANWORK": transwork,
        "DEGFIELD_ENCODED": degfield,
        "DEGFIELD2_ENCODED": degfield2,
        "SPEAKENG_ENCODED": speakeng,
        "EDUC_ENCODED": educ,
        "RACE": race,
        "BPL": bpl,
        "ANCESTR1": ancestr1,
        "OCCSOC": occsoc,  # Occupation code
        "IND": ind,        # Industry code
        "WKSWORK1": wkswork1
    }

    input_df = pd.DataFrame([input_dict])

    # Preprocess and predict
    input_transformed = preprocessor.transform(input_df)
    predicted_income = model.predict(input_transformed)[0]

    # Show prediction range
    lower = predicted_income - average_mae
    upper = predicted_income + average_mae

    st.subheader("Estimated Annual Income")
    st.success(f"${predicted_income:,.0f} (±${average_mae:,.0f})")
    st.write(f"**Range:** ${lower:,.0f} - ${upper:,.0f}")
