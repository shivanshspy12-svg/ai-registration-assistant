# AI-Powered Registration Assistant

A student-facing registration guidance prototype built with Python, Streamlit, and basic NLP.

## Features
- Chatbot for common registration questions
- TF-IDF + cosine similarity matching
- Sample eligibility checker
- Example course catalogue
- Guidance about registration documents and common portal issues

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the initial package installation

## Run locally

1. Extract the ZIP file.
2. Open a terminal in the project folder.
3. (Optional) Create a virtual environment:
   - Windows: `py -m venv .venv`
   - Activate: `.venv\Scripts\activate`
4. Install dependencies:
   `python -m pip install -r requirements.txt`
5. Start the app:
   `python -m streamlit run app.py`
6. Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Important customization
The FAQ list and course catalogue are sample content. The eligibility thresholds are illustrative and must be replaced with approved institutional rules before any real-world use. The demo does not authenticate students or submit course registrations.

## How the NLP works
1. Normalize the user's message (lowercase, remove punctuation, collapse spaces).
2. Convert the FAQ questions to TF-IDF vectors.
3. Convert the new question into the same vector space.
4. Use cosine similarity to find the closest FAQ.
5. Return the answer if the similarity is above a threshold; otherwise ask the user to rephrase.
