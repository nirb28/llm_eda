# LLM-Powered Dataset Analysis

An interactive web application that allows users to analyze datasets using Groq LLM through LangChain integration. Users can either upload a CSV file or provide a URL to a CSV dataset, then ask questions about the data in natural language.

## Features

- Upload CSV files or provide URLs to datasets
- Interactive data preview
- Natural language queries about your data
- Powered by Groq's Mixtral-8x7b model
- Beautiful Streamlit interface

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Copy `.env.example` to `.env`
   - Add your Groq API key to the `.env` file

## Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. Choose your data input method (file upload or URL)
2. Upload your CSV file or enter the URL
3. Preview your data
4. Ask questions about your dataset in natural language
5. Get AI-powered analysis and insights

## Requirements

- Python 3.8+
- Groq API key
- Internet connection for Groq API access