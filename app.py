import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI & Social Media Impact",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 AI & Social Media Impact")
st.subheader("Student Health Prediction")

st.write(
    "This Machine Learning application analyzes how social media usage, "
    "AI tool usage, sleep, physical activity, and demographic factors "
    "relate to student mental health."
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

file_path = "AI_SocialMedia_Student_Dataset.csv"

try:

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "srisyra02/ai-and-social-media-impact-student-health-and-grades",
        file_path,
    )

    st.success("✅ Dataset loaded successfully!")

except Exception as e:

    st.error(f"Unable to load dataset: {e}")
    st.stop()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("📋 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Students", df.shape[0])

with col2:
    st.metric("Features", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))


# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("🔍 View Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

st.header("🤖 Mental Health Prediction")

st.write(
    "The model predicts a student's Mental Health Score using "
    "social media usage, AI tool usage, sleep, physical activity, "
    "age, gender, and education level."
)


# Select features

features = [
    "Age",
    "Gender",
    "Education_Level",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Physical_Health_Score"
]

target = "Mental_Health_Score"


# Create model dataset

model_df = df[features + [target]].copy()


# Convert categorical columns

model_df = pd.get_dummies(
    model_df,
    columns=["Gender", "Education_Level"],
    drop_first=True
)


# Remove missing values

model_df = model_df.dropna()


X = model_df.drop(columns=[target])
y = model_df[target]


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Predictions

y_pred = model.predict(X_test)


# Evaluation

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with col2:

    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )


# --------------------------------------------------
# STUDENT PREDICTION
# --------------------------------------------------

st.header("🎯 Predict Student Mental Health Score")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        df["Gender"].dropna().unique()
    )

    education = st.selectbox(
        "Education Level",
        df["Education_Level"].dropna().unique()
    )

    social_media = st.number_input(
        "Daily Social Media Hours",
        min_value=0.0,
        max_value=24.0,
        value=3.0
    )


with col2:

    ai_usage = st.number_input(
        "Daily AI Tool Usage Hours",
        min_value=0.0,
        max_value=24.0,
        value=1.0
    )

    sleep = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    physical_activity = st.number_input(
        "Physical Activity Hours",
        min_value=0.0,
        max_value=24.0,
        value=1.0
    )

    physical_health = st.number_input(
        "Physical Health Score",
        min_value=float(df["Physical_Health_Score"].min()),
        max_value=float(df["Physical_Health_Score"].max()),
        value=float(df["Physical_Health_Score"].mean())
    )


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

if st.button("🔮 Predict Mental Health Score"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education_Level": [education],
        "Daily_Social_Media_Hours": [social_media],
        "Daily_AI_Tool_Usage_Hours": [ai_usage],
        "Sleep_Hours": [sleep],
        "Physical_Activity_Hours": [physical_activity],
        "Physical_Health_Score": [physical_health]
    })


    input_data = pd.get_dummies(
        input_data,
        columns=["Gender", "Education_Level"],
        drop_first=True
    )


    # Make sure input has same columns as training data

    input_data = input_data.reindex(
        columns=X.columns,
        fill_value=0
    )


    prediction = model.predict(input_data)[0]


    st.success(
        f"🧠 Predicted Mental Health Score: **{prediction:.2f}**"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI & Social Media Impact | Machine Learning Student Health Analysis"
)
