"""Exoplanet Finder — Streamlit entry point.

Run from the project root:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Exoplanet Finder", page_icon="🔭", layout="wide")

pages = [
    st.Page("pages/classify.py", title="Score a candidate", icon=":material/insights:", default=True),
    st.Page("pages/evaluation.py", title="Measured results", icon=":material/analytics:"),
    st.Page("pages/about.py", title="How it works & limits", icon=":material/menu_book:"),
]
st.navigation(pages).run()
