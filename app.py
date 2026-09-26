import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

model = load_model()

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🏖️",
    layout="wide"
)

st.title("🏖️ Wellness Tourism Package Purchase Predictor")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

# -----------------------------
# User Inputs
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Free Lancer",
            "Small Business",
            "Large Business"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    DurationOfPitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0,
        max_value=60,
        value=15
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Deluxe",
            "Standard",
            "Super Deluxe",
            "King"
        ]
    )


with col2:

    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

    NumberOfFollowups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=10,
        value=3
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star Rating",
        [3.0, 4.0, 5.0]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    NumberOfTrips = st.number_input(
        "Number of Trips per Year",
        min_value=0,
        max_value=20,
        value=2
    )

    Passport = st.selectbox(
        "Holds Passport?",
        ["No", "Yes"]
    )


with col3:

    PitchSatisfactionScore = st.slider(
        "Pitch Satisfaction Score",
        1,
        5,
        3
    )

    OwnCar = st.selectbox(
        "Owns a Car?",
        ["No", "Yes"]
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=5,
        value=0
    )

    Designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0,
        max_value=200000,
        value=20000
    )


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([{

        "Age": Age,

        "TypeofContact": TypeofContact,

        "CityTier": CityTier,

        "DurationOfPitch": DurationOfPitch,

        "Occupation": Occupation,

        "Gender": Gender,

        "NumberOfPersonVisiting":
            NumberOfPersonVisiting,

        "NumberOfFollowups":
            NumberOfFollowups,

        "ProductPitched":
            ProductPitched,

        "PreferredPropertyStar":
            PreferredPropertyStar,

        "MaritalStatus":
            MaritalStatus,

        "NumberOfTrips":
            NumberOfTrips,

        "Passport":
            1 if Passport == "Yes" else 0,

        "PitchSatisfactionScore":
            PitchSatisfactionScore,

        "OwnCar":
            1 if OwnCar == "Yes" else 0,

        "NumberOfChildrenVisiting":
            NumberOfChildrenVisiting,

        "Designation":
            Designation,

        "MonthlyIncome":
            MonthlyIncome

    }])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    if prediction == 1:

        st.success(
            f"✅ Likely to purchase the Wellness Tourism Package "
            f"(Probability: {probability:.1%})"
        )

    else:

        st.warning(
            f"❌ Unlikely to purchase the Wellness Tourism Package "
            f"(Probability: {probability:.1%})"
        )
