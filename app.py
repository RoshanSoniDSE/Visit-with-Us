import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🏖️",
    layout="wide"
)

# -------------------------------------------------
# TRAIN MODEL DIRECTLY FROM tourism.csv
# -------------------------------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv("tourism.csv")

    # Clean columns used in your notebook
    drop_cols = [
        c for c in ["Unnamed: 0", "CustomerID"]
        if c in df.columns
    ]

    df = df.drop(columns=drop_cols)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({
            "Fe Male": "Female"
        })

    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({
            "Unmarried": "Single"
        })

    df = df.drop_duplicates().reset_index(drop=True)

    target = "ProdTaken"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ])

    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=2,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    model.fit(X_train, y_train)

    return model


model = train_model()

# -------------------------------------------------
# USER INTERFACE
# -------------------------------------------------

st.title("🏖️ Wellness Tourism Package Purchase Predictor")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

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
        "Duration of Pitch",
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
        "Preferred Property Star",
        [3.0, 4.0, 5.0]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    NumberOfTrips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=20,
        value=2
    )

    Passport = st.selectbox(
        "Passport",
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
        "Owns Car?",
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


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([{

        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,

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
