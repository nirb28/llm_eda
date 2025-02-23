import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import requests
from io import StringIO

# Load environment variables
load_dotenv()

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

def load_data_from_url(url):
    """Load data from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error loading data from URL: {str(e)}")
        return None

def analyze_data(df, question):
    """Analyze the data using Groq LLM."""
    try:
        # Create a context about the data
        data_info = f"DataFrame Info:\n{df.info(buf=StringIO(), show_counts=True)}\n"
        data_head = f"First few rows:\n{df.head().to_string()}\n"
        
        # Construct the prompt
        prompt = f"""Given this dataset:
        {data_info}
        {data_head}
        
        Question: {question}
        
        Please provide a detailed analysis."""
        
        # Get response from Groq
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        return response.content
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

# Streamlit UI
st.title("Dataset Analysis")

# Check for API key
if not groq_api_key:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

# Data input method selection
data_input_method = st.radio("Choose data input method:", ["Upload CSV", "Enter URL"])

# Initialize session state for dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# Data upload/URL input
if data_input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
else:
    url = st.text_input("Enter the URL of your CSV file:")
    col1, col2 = st.columns([0.9, 0.1])
    sample_url = "https://raw.githubusercontent.com/nirb28/llm_eda/main/sample_sales_data.csv"
    with col1:
        st.markdown(f"*Try this sample dataset: [Sample Sales Data]({sample_url})*")
    with col2:
        if st.button("📋"):
            st.write("URL copied! ✅")
            st.session_state.url = sample_url
            url = sample_url
    if url:
        st.session_state.df = load_data_from_url(url)

# Display dataset if loaded
if st.session_state.df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df.head())
    
    # Question input
    question = st.text_area("What would you like to know about this dataset?")
    
    if st.button("Analyze"):
        if question:
            with st.spinner("Analyzing..."):
                analysis = analyze_data(st.session_state.df, question)
                if analysis:
                    st.subheader("Analysis")
                    st.write(analysis)
        else:
            st.warning("Please enter a question about the dataset.")
