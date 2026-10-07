import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Page configuration
st.set_page_config(
    page_title="AI & Social Media Impact",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 AI & Social Media Impact")
st.subheader("Student Health & Academic Performance")

st.write(
    "This application analyzes the relationship between "
    "social media usage, student health, and academic performance."
)

# Kaggle dataset file
file_path = "AI_SocialMedia_Student_Dataset.csv"

try:
    # Load dataset
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "srisyra02/ai-and-social-media-impact-student-health-and-grades",
        file_path,
    )

    st.success("✅ Dataset loaded successfully!")

    # Dataset overview
    st.write("## 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Dataset information
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Students", df.shape[0])

    with col2:
        st.metric("Number of Features", df.shape[1])

    # Columns
    st.write("## 🔎 Dataset Columns")
    st.write(list(df.columns))

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
