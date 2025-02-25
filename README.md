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

## Sample Questions
Here are some example questions you can ask about this dataset:

Basic Analysis Questions:

"What is the total revenue from all sales?"
"Which product category has the highest average price?"
"What is the most popular product based on quantity sold?"

Regional Analysis:

"How do sales vary across different regions?"
"Which region has the highest total revenue?"
"What's the most popular product category in each region?"
Customer Demographics:

"What's the average customer age for electronics purchases?"
"Is there a correlation between customer age and purchase amount?"
"Which age group buys the most expensive products?"
Temporal Analysis:

"What was the daily sales trend?"
"Which day had the highest total sales?"
"Is there a pattern in purchase quantities over time?"
Product-specific Analysis:

"What's the revenue breakdown by product category?"
"Which products have above-average sales quantities?"
"What's the price range distribution across different categories?"
You can also ask more complex questions that combine multiple aspects:

"What's the relationship between customer age, product category, and average purchase amount?"
"Are there any notable patterns in how different age groups shop across regions?"
"Which product categories perform best in each region, considering both quantity and revenue?"