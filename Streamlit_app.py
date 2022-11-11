import streamlit as st
import pandas as pd
#import geopandas as gpd
import altair as alt




st.set_page_config(layout="wide")
st.title("Interact with Gapminder Data")

#df = pd.read_csv("G:/My Drive/MADS course transcripts/699 Capstone/coi_district_grouped.csv")
#df_gdp_o = df.query("continent=='Oceania' & metric=='gdpPercap'")

title = "GDP for countries in Oceania"
#fig = px.line(df, x = "year", y = "NAME_LEA15", color = "year", title = title)
#st.plotly_chart(fig, use_container_width=True)


#test = gpd.read_file('schooldistrict_sy1314_tl15.shp')
#chart = alt.Chart(test).mark_geoshape()
#st.altair_chart(chart)

from gsheetsdb import connect

# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

sheet_url = st.secrets["public_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

# Print results.
for row in rows:
    st.write(f"{row.name} has a :{row.pet}:")