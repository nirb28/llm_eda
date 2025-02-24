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
from sqlalchemy import create_engine, inspect
import urllib.parse
from pathlib import Path

# Get the application's base directory
BASE_DIR = Path(__file__).parent.absolute()

# Load environment variables
load_dotenv()

# Available Groq models
GROQ_MODELS = {
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Deepseek R1 / Llama 70B": "deepseek-r1-distill-llama-70b",
    "Gemma 9B": "gemma2-9b-it",
}

# Database types and their connection string formats
DB_TYPES = {
    "PostgreSQL": "postgresql://{user}:{password}@{host}:{port}/{database}",
    "MySQL": "mysql+pymysql://{user}:{password}@{host}:{port}/{database}",
    "SQLite": "sqlite:///{database_path}",
}

# Initialize session states
if 'model_name' not in st.session_state:
    st.session_state.model_name = "deepseek-r1-distill-llama-70b"
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'selected_table' not in st.session_state:
    st.session_state.selected_table = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

def get_api_key():
    """Get API key from various sources."""
    # Try environment variable first
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key
    
    # Try streamlit secrets
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    # If no API key found, let user input it
    return st.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Your API key will not be stored permanently"
    )

def initialize_llm():
    """Initialize or reinitialize the Groq LLM with selected model."""
    groq_api_key = get_api_key()
    
    if not groq_api_key:
        st.error("Please provide your Groq API Key")
        st.stop()
    
    st.session_state.llm = ChatGroq(
        api_key=groq_api_key,
        model_name=st.session_state.model_name
    )
    return st.session_state.llm

# Initialize Groq LLM at startup
initialize_llm()

def connect_to_database(db_type, **params):
    """Create database connection using SQLAlchemy."""
    try:
        if db_type == "SQLite":
            # Handle relative paths for SQLite
            db_path = params['database_path']
            if not os.path.isabs(db_path):
                db_path = os.path.join(BASE_DIR, db_path)
            conn_str = DB_TYPES[db_type].format(database_path=db_path)
        else:
            # URL encode the password to handle special characters
            params['password'] = urllib.parse.quote_plus(params['password'])
            conn_str = DB_TYPES[db_type].format(**params)
        
        engine = create_engine(conn_str)
        # Test the connection
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def get_table_names(engine):
    """Get list of tables in the database."""
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        st.error(f"Error getting tables: {str(e)}")
        return []

def load_table_data(engine, table_name):
    """Load data from a database table into a pandas DataFrame."""
    try:
        return pd.read_sql_table(table_name, engine)
    except Exception as e:
        st.error(f"Error loading table data: {str(e)}")
        return None

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
        response = st.session_state.llm.invoke(messages)
        
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
st.title("Dataset Analysis")

# Model selection in sidebar
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
    initialize_llm()
    st.sidebar.success(f"Model updated to {selected_model}")

# Add model description
model_descriptions = {
    "Mixtral 8x7B": "A powerful mixture-of-experts model with strong analytical capabilities.",
    "Deepseek R1 / Llama 70B": "Deepseek R1, excellent for complex reasoning.",
    "Gemma 9B": "Google's efficient model, good balance of performance and speed."
}
st.sidebar.markdown(f"*{model_descriptions[selected_model]}*")

# Data input method selection
data_input_method = st.radio("Choose data input method:", ["Upload CSV", "Enter URL", "Connect to Database"])

# Initialize session state for dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# Data input handling
if data_input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
elif data_input_method == "Enter URL":
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
else:  # Database connection
    db_type = st.selectbox("Select Database Type", options=list(DB_TYPES.keys()))
    
    # Database connection parameters
    if db_type == "SQLite":
        db_params = {
            'database_path': st.text_input("Database Path", 
                help="You can use relative paths like 'sample_movies.db' or absolute paths")
        }
        st.markdown("*Try the sample database: `sample_movies.db`*")
    else:
        col1, col2 = st.columns(2)
        with col1:
            db_params = {
                'host': st.text_input("Host", "localhost"),
                'port': st.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306"),
                'database': st.text_input("Database Name"),
            }
        with col2:
            db_params.update({
                'user': st.text_input("Username"),
                'password': st.text_input("Password", type="password"),
            })
    
    # Connect button
    if st.button("Connect to Database"):
        engine = connect_to_database(db_type, **db_params)
        if engine:
            st.session_state.db_engine = engine
            st.session_state.tables = get_table_names(engine)
            st.success("Successfully connected to the database!")
    
    # Table selection if connected
    if st.session_state.db_engine and st.session_state.tables:
        selected_table = st.selectbox("Select Table", options=st.session_state.tables)
        if selected_table != st.session_state.selected_table:
            st.session_state.selected_table = selected_table
            st.session_state.df = load_table_data(st.session_state.db_engine, selected_table)

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
