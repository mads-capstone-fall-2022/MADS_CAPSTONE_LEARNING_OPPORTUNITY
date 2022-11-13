#import geopandas as gpd
import altair as alt
import pandas as pd
import streamlit as st
from gsheetsdb import connect

st.set_page_config(layout="wide")
st.title("Dashboard for Child oppurtunity index!")

title = "GDP for countries in Oceania"


# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

sheet_url = st.secrets["troy_test_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')
#print(type(rows))
#st.map(data=rows, zoom=None, use_container_width=True)
st.table(rows)
