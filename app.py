import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import requests
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import json
import re

# Load environment variables
load_dotenv()

# Available Groq models
GROQ_MODELS = {
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Deepseek R1 / Llama 70B": "deepseek-r1-distill-llama-70b",
    "Gemma 9B": "gemma2-9b-it",
}

# Initialize session state for model
if 'model_name' not in st.session_state:
    st.session_state.model_name = "deepseek-r1-distill-llama-70b"

def initialize_llm():
    """Initialize or reinitialize the Groq LLM with selected model."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        api_key=groq_api_key,
        model_name=st.session_state.model_name
    )

# Initialize Groq LLM
llm = initialize_llm()

def create_chart(df, chart_type, x, y, title, color=None):
    """Create a Plotly chart based on specifications."""
    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, title=title, color=color)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, title=title, color=color)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title, color=color)
        elif chart_type == "pie":
            fig = px.pie(df, values=y, names=x, title=title)
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, title=title, color=color)
        else:
            return None
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def load_data_from_url(url):
    """Load data from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error loading data from URL: {str(e)}")
        return None

def extract_json_from_text(text):
    """Extract JSON from text, handling various formats."""
    try:
        # Find any text that looks like a JSON object with "charts"
        json_pattern = r'(?:"charts"|\'charts\')\s*:\s*\[(.*?)\]'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            charts_content = match.group(1)
            # Clean up the content and make it valid JSON
            charts_content = charts_content.strip()
            if not charts_content.endswith(']'):
                charts_content += ']'
            if not charts_content.startswith('['):
                charts_content = '[' + charts_content
                
            # Wrap in a proper JSON object
            json_str = '{"charts": ' + charts_content + '}'
            
            # Handle single quotes and normalize whitespace
            json_str = json_str.replace("'", '"').replace('\n', ' ')
            
            return json.loads(json_str)
    except Exception as e:
        st.error(f"Error parsing chart JSON: {str(e)}")
        return None

def analyze_data(df, question):
    """Analyze the data using Groq LLM."""
    try:
        # Create a context about the data
        data_info = f"DataFrame Info:\n{df.info(buf=StringIO(), show_counts=True)}\n"
        data_head = f"First few rows:\n{df.head().to_string()}\n"
        data_describe = f"Numerical Summary:\n{df.describe().to_string()}\n"
        
        # Construct the prompt
        prompt = f"""Given this dataset:
        {data_info}
        {data_head}
        {data_describe}
        
        Question: {question}
        
        Please provide:
        1. A detailed analysis of the data
        2. If relevant, suggest one or more visualizations with these specifications:
           - Chart type (one of: bar, line, scatter, pie, box)
           - X-axis column
           - Y-axis column
           - Title
           - Color column (optional)
        
        Format visualization suggestions as JSON like this:
        "charts": [
            {{"type": "bar", "x": "column1", "y": "column2", "title": "Chart Title", "color": "column3"}}
        ]
        
        Only include the JSON if visualizations would be helpful for the analysis.
        Make sure to format the JSON exactly as shown, with double quotes around keys and string values.
        """
        
        # Get response from Groq
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        # Try to extract any chart specifications from the response
        charts_data = extract_json_from_text(response.content)
        
        if charts_data and 'charts' in charts_data:
            st.subheader("Visualizations")
            for chart_spec in charts_data['charts']:
                try:
                    # Verify required fields are present
                    required_fields = ['type', 'x', 'y', 'title']
                    if not all(field in chart_spec for field in required_fields):
                        st.warning(f"Skipping chart due to missing required fields. Got: {chart_spec}")
                        continue
                    
                    fig = create_chart(
                        df,
                        chart_spec['type'],
                        chart_spec['x'],
                        chart_spec['y'],
                        chart_spec['title'],
                        chart_spec.get('color')  # color is optional
                    )
                    if fig:
                        st.plotly_chart(fig)
                    else:
                        st.warning(f"Could not create chart with specification: {chart_spec}")
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
                    st.code(chart_spec, language="json")
        
        # Return the text analysis
        return response.content
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

# Streamlit UI
st.title("Dataset Analysis with Groq LLM")

# Check for API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

# Model selection
st.sidebar.header("Model Settings")
selected_model = st.sidebar.selectbox(
    "Choose Groq Model",
    options=list(GROQ_MODELS.keys()),
    format_func=lambda x: x,
    index=list(GROQ_MODELS.values()).index(st.session_state.model_name)
)

# Update model if changed
if GROQ_MODELS[selected_model] != st.session_state.model_name:
    st.session_state.model_name = GROQ_MODELS[selected_model]
    llm = initialize_llm()
    st.sidebar.success(f"Model updated to {selected_model}")

# Add model description
model_descriptions = {
    "Mixtral 8x7B": "A powerful mixture-of-experts model with strong analytical capabilities.",
    "Deepseek R1 / Llama 70B": "Deepseek R1, excellent for complex reasoning.",
    "Gemma 9B": "Google's efficient model, good balance of performance and speed."
}
st.sidebar.markdown(f"*{model_descriptions[selected_model]}*")

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
