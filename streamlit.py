import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ MUST come before any Streamlit command
st.set_page_config(page_title="Income Prediction App", layout="wide")

# Example of manual mappings for selected features
state_name_to_fips = {
        "Alabama": 1, "Alaska": 2, "Arizona": 4, "Arkansas": 5,
        "California": 6, "Colorado": 8, "Connecticut": 9, "Delaware": 10,
        "District of Columbia": 11, "Florida": 12, "Georgia": 13,
        "Hawaii": 15, "Idaho": 16, "Illinois": 17, "Indiana": 18,
        "Iowa": 19, "Kansas": 20, "Kentucky": 21, "Louisiana": 22,
        "Maine": 23, "Maryland": 24, "Massachusetts": 25, "Michigan": 26,
        "Minnesota": 27, "Mississippi": 28, "Missouri": 29, "Montana": 30,
        "Nebraska": 31, "Nevada": 32, "New Hampshire": 33, "New Jersey": 34,
        "New Mexico": 35, "New York": 36, "North Carolina": 37,
        "North Dakota": 38, "Ohio": 39, "Oklahoma": 40, "Oregon": 41,
        "Pennsylvania": 42, "Rhode Island": 44, "South Carolina": 45,
        "South Dakota": 46, "Tennessee": 47, "Texas": 48, "Utah": 49,
        "Vermont": 50, "Virginia": 51, "Washington": 53,
        "West Virginia": 54, "Wisconsin": 55, "Wyoming": 56
}
sex_map = {'Male': 0, 'Female': 1}
REGION_map = {
        'East South Central Div.': 1,
        'Pacific Division': 2,
        'Mountain Division': 3,
        'West South Central Div.': 4, 
        'New England Division': 5,
        'South Atlantic Division': 6, 
        'East North Central Div.': 7,
        'West North Central Div.': 8,
        'Middle Atlantic Division': 9
    }
MARST_map = {
        'Never married/single': 1,
        'Married, spouse present': 2, 
        'Divorced' : 3,
        'Separated': 4,
        'Widowed': 5,
        'Married': 6,
        'spouse absent': 7
    }
RACE_map = {
        'Black/African American': 1, 
        'White': 2, 
        'Two major races': 3, 
        'Other race, nec': 4,
        'Other Asian or Pacific Islander': 5, 
        'American Indian or Alaska Native': 6,
        'Chinese': 7, 
        'Three or more major races': 8,
        'Japanese': 9
    }
BPL_map = {
        'Georgia': 1, 'Alabama': 2, 'Florida': 3, 'Missouri': 4, 'Mexico': 5, 'New York': 6, 'California': 7, 'New Jersey': 8, 'North Carolina': 9, 'Nevada': 10,
    'India': 11, 'Michigan': 12, 'Maryland': 13, 'Pennsylvania': 14, 'Germany': 15, 'Indiana': 16, 'Illinois': 17, 'Colorado': 18, 'Tennessee': 19, 'Mississippi': 20,
    'Cuba': 21, 'Kentucky': 22, 'China': 23, 'West Indies': 24, 'Texas': 25, 'Central America': 26, 'New Hampshire': 27, 'SOUTH AMERICA': 28, 'Ohio': 29, 'Wisconsin': 30,
    'Oklahoma': 31, 'Washington': 32, 'Belgium': 33, 'Nebraska': 34, 'Maine': 35, 'Other USSR/Russia': 36, 'Iraq': 37, 'Massachusetts': 38, 'Virginia': 39, 'Alaska': 40,
    'Rhode Island': 41, 'Pacific Islands': 42, 'Louisiana': 43, 'South Carolina': 44, 'West Virginia': 45, 'Arizona': 46, 'Korea': 47, 'France': 48, 'Idaho': 49, 'Connecticut': 50,
    'Canada': 51, 'Lebanon': 52, 'District of Columbia': 53, 'Kansas': 54, 'Vietnam': 55, 'Puerto Rico': 56, 'Minnesota': 57, 'Oregon': 58, 'Guam': 59, 'Hawaii': 60,
    'Australia and New Zealand': 61, 'Delaware': 62, 'Arkansas': 63, 'Wyoming': 64, 'United Kingdom, ns': 65, 'Japan': 66, 'Philippines': 67, 'Iowa': 68, 'South Dakota': 69,
    'Italy': 70, 'New Mexico': 71, 'Nepal': 72, 'AFRICA': 73, 'Montana': 74, 'Utah': 75, 'North Dakota': 76, 'Laos': 77, 'Vermont': 78, 'Spain': 79, 'Malaysia': 80,
    'England': 81, 'Syria': 82, 'Greece': 83, 'Indonesia': 84, 'Turkey': 85, 'Thailand': 86, 'Yugoslavia': 87, 'Romania': 88, 'Europe, ns': 89, 'Poland': 90, 'Austria': 91,
    'Iran': 92, 'Netherlands': 93, 'Ireland': 94, 'Cambodia (Kampuchea)': 95, 'Saudi Arabia': 96, 'Singapore': 97, 'Hungary': 98, 'Portugal': 99, 'United Arab Emirates': 100,
    'Asia, nec/ns': 101, 'Kuwait': 102, 'Other n.e.c.': 103, 'Iceland': 104, 'Afghanistan': 105, 'Switzerland': 106, 'U.S. Virgin Islands': 107, 'Israel/Palestine': 108,
    'Atlantic Islands': 109, 'Scotland': 110, 'Lithuania': 111, 'Bulgaria': 112, 'Finland': 113, 'Albania': 114, 'Czechoslovakia': 115, 'Americas, n.s.': 116, 'Jordan': 117,
    'Sweden': 118, 'Denmark': 119, 'Latvia': 120, 'Yemen Arab Republic (North)': 121, 'American Samoa': 122, 'Norway': 123
    }
ANCESTR1_map = {
        'African-American': 1, 'Not Reported': 2, 'White/Caucasian': 3, 'Italian': 4, 'Mexican': 5, 'German': 6, 'Afro-American': 7, 'English': 8, 'Irish, various subheads,': 9,
    'United States': 10, 'European, nec': 11, 'Welsh': 12, 'French': 13, 'Spanish': 14, 'Asian Indian': 15, 'Scotch Irish': 16, 'Norwegian': 17, 'Uncodable': 18, 'Spaniard': 19,
    'Polish': 20, 'Chinese': 21, 'American Indian  (all tribes)': 22, 'Other Pacific': 23, 'French Canadian': 24, 'Scottish': 25, 'African': 26, 'Colombian': 27, 'Southern European, nec': 28,
    'Hungarian': 29, 'Swedish': 30, 'British': 31, 'Hispanic': 32, 'Ukrainian': 33, 'Kurdish': 34, 'Peruvian': 35, 'Puerto Rican': 36, 'Korean': 37, 'Slovak': 38, 'Greek': 39,
    'Mixture': 40, 'Austrian': 41, 'Scandinavian, Nordic': 42, 'Cuban': 43, 'Lebanese': 44, 'Swiss': 45, 'Vietnamese': 46, 'Dutch': 47, 'Chilean': 48, 'Australian': 49,
    'Belgian': 50, 'Russian': 51, 'Northern European, nec': 52, 'Hawaiian': 53, 'Brazilian': 54, 'Taiwanese': 55, 'Japanese': 56, 'Ecuadorian': 57, 'Yugoslavian': 58, 'Canadian': 59,
    'Guatemalan': 60, 'Honduran': 61, 'Western European, nec': 62, 'Salvadoran': 63, 'Pakistani': 64, 'Venezuelan': 65, 'Nuevo Mexicano': 66, 'Filipino': 67, 'Asian': 68, 'Danish': 69,
    'Jamaican': 70, 'Portuguese': 71, 'Panamanian': 72, 'Estonian': 73, 'Palestinian': 74, 'Mexican American': 75, 'Bulgarian': 76, 'Iranian': 77, 'Romanian': 78, 'Haitian': 79,
    'Egyptian': 80, 'Nepali': 81, 'Latvian': 82, 'Czechoslovakian': 83, 'Samoan': 84, 'Serbian': 85, 'Acadian': 86, 'Algerian': 87, 'Nigerian': 88, 'Eastern European, nec': 89,
    'Other Subsaharan Africa': 90, 'Jordanian': 91, 'Cameroonian': 92, 'Polynesian': 93, 'Syrian': 94, 'Latin American': 95, 'Slav': 96, 'Chamorro Islander': 97, 'Dominican': 98,
    'West Indian': 99, 'Armenian': 100, 'North American': 101, 'Trinidadian/Tobagonian': 102, 'Guamanian': 103, 'Lithuanian': 104, 'Indonesian': 105, 'Pacific Islander': 106,
    'Eskimo': 107, 'Slovene': 108, 'Other': 109, 'Finnish': 110, 'Zimbabwean': 111, 'New Zealander': 112, 'Mongolian': 113, 'Sicilian': 114, 'Laotian': 115, 'British Isles': 116,
    'Togo': 117, 'Assyrian/Chaldean/Syriac': 118, 'Liberian': 119, 'Iraqi': 120, 'Croatian': 121, 'Guyanese/British Guiana': 122, 'Other Asian': 123, 'Arab': 124, 'Basque': 125,
    'Marshall Islander': 126, 'South American': 127, 'Bengali': 128, 'Cambodian': 129, 'Malaysian': 130, 'Ghanian': 131, 'Congolese': 132, 'Ethiopian': 133, 'Argentinean': 134,
    'Tongan': 135, 'Kenyan': 136, 'Israeli': 137, 'Chicano/Chicana': 138, 'Burmese': 139, 'Hong Kong': 140, 'Thai': 141, 'Bohemian': 142, 'Belorussian': 143, 'Prussian': 144,
    'Icelander': 145, 'Afghan': 146, 'Eritrean': 147, 'West African': 148, 'Sri Lankan': 149, 'Macedonian': 150, 'Turkish': 151, 'South African': 152, 'Nicaraguan': 153,
    'Middle Eastern': 154, 'Germans from Russia': 155, 'Somalian': 156, 'Bolivian': 157, 'Sierra Leonean': 158, 'Spanish American': 159, 'Maltese': 160, 'Luxemburger': 161,
    'Moroccan': 162, 'Cantonese': 163, 'Bahamian': 164, 'Senegalese': 165, 'Albanian': 166, 'Other Arab': 167, 'Dutch West Indies': 168, 'Micronesian': 169, 'Hmong': 170,
    'Costa Rican': 171, 'Fijian': 172, 'Saudi Arabian': 173, 'Punjabi': 174, 'Moldavian': 175, 'Sudanese': 176, 'South American Indian': 177, 'Uruguayan': 178, 'Belizean': 179,
    'Central American Indian': 180, 'Flemish': 181, 'Georgian': 182, 'St Lucia Islander': 183, 'Yemeni': 184, 'Cape Verdean': 185, 'Barbadian': 186, 'Paraguayan': 187,
    'Central European, nec': 188, 'Libyan': 189, 'Okinawan': 190, 'Texas': 191, 'Other West Indian': 192, 'Tibetan': 193, 'Ugandan': 194, 'Cossack': 195, 'Uzbek': 196,
    'North African': 197, 'Rom': 198, 'British West Indian': 199, 'British Virgin Islander': 200, 'Anguilla Islander': 201, 'Grenadian': 202, 'Alsatian, Alsace-Lorraine': 203,
    'Gambian': 204, 'Bhutanese': 205, 'Guinean': 206
    }
LANGUAGE_map = {
    'English': 1, 'Spanish': 2, 'Dravidian': 3, 'German': 4, 'Hindi and related': 5, 'Japanese': 6, 'French': 7, 'Ukrainian, Ruthenian, Little Russian': 8, 'Other Persian dialects': 9,
    'Korean': 10, 'Greek': 11, 'Polish': 12, 'Arabic': 13, 'Chinese': 14, 'Vietnamese': 15, 'Italian': 16, 'Filipino, Tagalog': 17, 'Sub-Saharan Africa': 18, 'Tibetan': 19,
    'Portuguese': 20, 'Russian': 21, 'Dutch': 22, 'Indonesian': 23, 'Native': 24, 'Serbo-Croatian, Yugoslavian, Slavonian': 25, 'Rumanian': 26, 'Aleut, Eskimo': 27, 'Persian, Iranian, Farsi': 28,
    'Micronesian, Polynesian': 29, 'Thai, Siamese, Lao': 30, 'Navajo': 31, 'Aztecan, Nahuatl, Uto-Aztecan': 32, 'Near East Arabic dialect': 33, 'Athapascan': 34, 'Hebrew, Israeli': 35,
    'Other East/Southeast Asian': 36, 'Amharic, Ethiopian, etc.': 37, 'Burmese, Lisu, Lolo': 38, 'Czech': 39, 'Turkish': 40, 'Magyar, Hungarian': 41, 'Hamitic': 42, 'Other Balto-Slavic': 43,
    'Finnish': 44, 'Other Afro-Asiatic languages': 45, 'Hawaiian': 46, 'Iroquoian': 47, 'Armenian': 48, 'Other Altaic': 49, 'Other Malayan': 50, 'Celtic': 51, 'Lithuanian': 52,
    'Swedish': 53, 'Albanian': 54, 'Yiddish, Jewish': 55, 'Other or not reported': 56, 'Algonquian': 57, 'Norwegian': 58, 'Danish': 59, 'Slovak': 60, 'Muskogean': 61, 'Siouan languages': 62,
    'Keres': 63
    }
TRANWORK_map = {
    'Auto, truck, or van': 1, 'Walked only': 2, 'Worked at home': 3, 'Bus': 4, 'Bicycle': 5, 'Other': 6, 'Motorcycle': 7, 'Taxicab': 8, 'Ferryboat': 9,
    'Light rail, streetcar, or trolley (Carro público in PR)': 10, 'Subway or elevated': 11, 'Long-distance train or commuter train': 12
    }


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

st.title("Predicted Personal Income")

# Define input form
with st.form("income_form"):
    st.write("### Enter Person's Information")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 0, 120, 30)
        sex = st.selectbox("Sex", list(sex_map.keys()))
        sexcode = sex_map[sex]
        state_name = st.selectbox("State", list(state_name_to_fips.keys()))
        statefip = state_name_to_fips[state_name]
        region = st.selectbox("Region", list(REGION_map.keys()))
        region_code = REGION_map[region]
        marital_status = st.selectbox("Marital Status (MARST)", list(MARST_map.keys()))
        marital_status_code = MARST_map[marital_status]
        nchil = st.number_input("Number of Children", 0, 9, 0)
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
        race = st.number_input("Race", list(RACE_map.keys()))
        race_code = RACE_map[race]
        bpl = st.number_input("Birthplace Code (BPL)", 1, 999, 100)
        ancestr1 = st.number_input("Ancestry Code", 0, 999, 100)
        language = st.number_input("Language", 0, 999, 100)
        perwt = st.number_input("Person Weight", 1, 999, 100)
        
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
        "SEX": sexcode,
        "STATEFIP": statefip,
        "REGION": region_code,
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
        "RACE": race_code,
        "BPL": bpl,
        "ANCESTR1": ancestr1,
        "LANGUAGE": language,
        "OCCSOC": occsoc,  # Occupation code
        "IND": ind,        # Industry code
        "PERWT": perwt,
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
