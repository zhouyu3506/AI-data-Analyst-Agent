An end-to-end AI-powered data analysis dashboard built with:

- **Streamlit** for UI
- **BigQuery** for analytics
- **LLM (OpenAI)** for:
  - Natural Language → SQL
  - Data → Business Summary

## Features
- Ask business questions in natural language
- Automatically generate BigQuery SQL
- Execute queries and visualize trends
- Generate concise business insights using LLM

## Tech Stack
- Python
- Streamlit
- Google BigQuery
- OpenAI API
- Docker
- Google Cloud Run

## Architecture
User Intent
↓
LLM → SQL
↓
BigQuery
↓
DataFrame → Chart
↓
LLM Summary
