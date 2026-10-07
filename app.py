import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter

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

# Load Kaggle dataset
file_path = ""

try:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "srisyra02/ai-and-social-media-impact-student-health-and-grades",
        file_path,
    )

    st.success("Dataset loaded successfully!")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Information")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    st.write("### Columns")
    st.write(list(df.columns))

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
