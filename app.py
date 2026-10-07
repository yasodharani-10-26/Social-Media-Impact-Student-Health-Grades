import streamlit as st
import kagglehub
import os

st.set_page_config(
    page_title="AI & Social Media Impact",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI & Social Media Impact")
st.subheader("Student Health & Academic Performance")

st.write(
    "This application analyzes the relationship between "
    "social media usage, student health, and academic performance."
)

try:
    # Download the Kaggle dataset
    dataset_path = kagglehub.dataset_download(
        "srisyra02/ai-and-social-media-impact-student-health-and-grades"
    )

    st.success("Dataset downloaded successfully!")

    st.write("### Dataset location")
    st.write(dataset_path)

    st.write("### Files available in the dataset")

    files = os.listdir(dataset_path)

    for file in files:
        st.write("📄", file)

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
