# Zepto Data & AI Platform - Capstone Project

## Project Overview
This repository contains three internally-linked modules for the Zepto platform: a data engineering pipeline, an analytics pipeline, and a GenAI support assistant.

## Setup Instructions
1. Clone the repository.
2. Install the dependencies:
   pip install -r requirements.txt

## Module 1: Data Pipeline (/data_pipeline)
Run the script: python data_pipeline/pipeline.py
- Description: Scrapes book data from books.toscrape.com, cleans it, and converts GBP to INR using the fixed project-defined rate: 1 GBP = 105.50 INR.
- Output: Stores data in an SQLite database (zepto_books.db) with a normalized two-table schema (categories and books).

## Module 2: Analytics (/analytics)
Run the EDA: python analytics/01_eda.py
Run the modeling: python analytics/02_modeling.py
- Description: End-to-end EDA and predictive modeling pipeline on the Titanic dataset.
- Missing Values: Handled missing values based on percentage thresholds (under 5% dropped, 5-30% imputed with median/mode, over 30% dropped or encoded).
- Modeling: Trains Logistic Regression, Decision Tree, and Random Forest models. Evaluates them using accuracy, precision, recall, F1, and ROC-AUC. Handles class imbalance using SMOTE and tunes Random Forest with GridSearchCV.
- Output: Saves the best full pipeline (best_titanic_pipeline.pkl) using joblib.

## Module 3: Support Assistant (/support_assistant)
Run locally: uvicorn main:app --host 0.0.0.0 --port 7860
Run with Docker: docker build -t zepto-assistant . then docker run -p 7860:7860 zepto-assistant
- Description: A LangGraph-based RAG support assistant. It ingests 8 Zepto policy documents, embeds them locally using all-MiniLM-L6-v2, and stores them in ChromaDB.
- Architecture: Ingestion (docs/) -> Embedding (sentence-transformers) -> Retrieval (ChromaDB) -> Generation (LangGraph + FastAPI mock LLM). The pipeline uses a MOCK_LLM environment variable for the deterministic graded baseline.
- Output: Exposes a POST /ask endpoint that returns structured JSON (answer, sources, confidence).
