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

sheet_url = st.secrets["child_oppurtunity_input_file"] #st.secrets["seda_map_file"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')


#print(type(rows))
#seda_map_df  = pd.DataFrame(list(rows))
child_oppurtunity_df = pd.DataFrame(list(rows))

#add filters
v_segment = st.selectbox(
     'Which segment would you like to select',
     ('Segment 1', 'Segment 2', 'Segment 3','Segment 4'))
v_year_choice = st.sidebar.slider(
    'Year:', min_value=2010, max_value=2015, step=1, value=2015)


#filter the dataframe
child_oppurtunity_df = child_oppurtunity_df[child_oppurtunity_df['year']. == v_year_choice]
child_oppurtunity_df = child_oppurtunity_df[child_oppurtunity_df['segment'].isin(v_segment)]
    
st.set_page_config(layout="wide")
#fig = make_subplots(rows=1, cols=2)

#fig = px.scatter_mapbox(seda_map_df, lat="latitude", lon="longitude", hover_name="NAME", hover_data=["GEOID"],
                    #    color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig = px.scatter_mapbox(child_oppurtunity_df,lat='latitude', lon='longitude',color_continuous_scale = 'rdylgn', color='Segment',color_discrete_sequence=px.colors.qualitative.G10, size='cs_mn_all_abs' ,text='sedaleaname')#, hover_name='subject')
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_geos(fitbounds="locations")
fig.update_layout(mapbox_bounds={"west": -180, "east": -50, "south": 20, "north": 90})
st.plotly_chart(fig, use_container_width=True)
#fig.show()
#st.map(data=seda_map_df, zoom=None, use_container_width=True)
#st.table(rows)
