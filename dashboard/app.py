import streamlit as st
import pandas as pd

st.title("DM Bypass Dashboard (MVP)")
team = st.text_input("Team name (exact)", "Manchester United")
st.write("Drop precomputed CSVs into `reports/` and we’ll visualize them here.")

try:
    summary = pd.read_csv("reports/bypass_summary.csv")
    st.line_chart(summary.set_index("match_date")["bypass_rate"])
except Exception as e:
    st.info("No reports yet. Run the pipeline to generate artifacts.")
