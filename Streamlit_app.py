#import geopandas as gpd
import altair as alt
import pandas as pd
import streamlit as st
from gsheetsdb import connect
import plotly.express as px
from plotly.subplots import make_subplots

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

sheet_url = st.secrets["seda_map_file"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')
#print(type(rows))
seda_map_df  = pd.DataFrame(list(rows))

fig = make_subplots(rows=1, cols=2)

fig = px.scatter_mapbox(seda_map_df, lat="latitude", lon="longitude", hover_name="NAME", hover_data=["GEOID"],
                        color_discrete_sequence=["fuchsia"], zoom=3, height=300,
                        row=1, col=1)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(mapbox_bounds={"west": -180, "east": -50, "south": 20, "north": 90})
st.plotly_chart(fig, use_container_width=True)
#fig.show()
#st.map(data=seda_map_df, zoom=None, use_container_width=True)
#st.table(rows)
