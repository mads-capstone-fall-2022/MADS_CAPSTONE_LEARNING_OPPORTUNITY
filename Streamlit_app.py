#import geopandas as gpd
import altair as alt
import pandas as pd
import streamlit as st
from gsheetsdb import connect
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
Report, Dashboard = st.tabs(["Report Page", "Dashboard Page"])
Dashboard.title("Compare Achievment scores on the same Scale!")




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
v_segment = Dashboard.sidebar.selectbox(
     'Which segment would you like to select',
     ('Segment 1', 'Segment 2', 'Segment 3','Segment 4'))

v_subject = Dashboard.sidebar.selectbox(
     'Which subject would you like to select',
     ('Math', 'Reading'))

v_year_choice = Dashboard.sidebar.slider(
    'Year:', min_value=2010, max_value=2015, step=1, value=2015)


#filter the dataframe
child_oppurtunity_df = child_oppurtunity_df[child_oppurtunity_df['year'] == v_year_choice]
child_oppurtunity_df = child_oppurtunity_df[child_oppurtunity_df['Segment']==int(v_segment[-1])-1]
child_oppurtunity_df = child_oppurtunity_df[child_oppurtunity_df['subject']==v_subject]   
#st.set_page_config(layout="wide")
#fig = make_subplots(rows=1, cols=2)

#fig = px.scatter_mapbox(seda_map_df, lat="latitude", lon="longitude", hover_name="NAME", hover_data=["GEOID"],
                    #    color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig = px.scatter_mapbox(child_oppurtunity_df,lat='latitude', lon='longitude', zoom=3, height=300, hover_name='sedaleaname',color_continuous_scale = 'rdylgn', color='subject', size='cs_mn_all_abs' ,text='sedaleaname')#, hover_name='subject')
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_geos(fitbounds="locations")
fig.update_layout(mapbox_bounds={"west": -180, "east": -50, "south": 20, "north": 90})
Dashboard.plotly_chart(fig, use_container_width=True)
#fig.show()
#st.map(data=seda_map_df, zoom=None, use_container_width=True)
#st.table(rows)

Report.header('Team Learning Opportunity Blog Post')
Report.text('Jay Korrapati and Katie Andrews')
Report.subheader('Introduction')

Report.markdown(f'''When school districts in the US are judged, it is usually by comparison to other districts.  Parents use ratings sites like GreatSchools - which uses test scores, graduation rates, and other data (GreatSchools.org, n.d.) - to compare schools when they are looking to move to a new area.  State governments use standardized test scores to rank schools and districts and identify struggling schools (Klein, 2015).  The standardized test scores used in both cases were designed at the state level in response to the 2001 No Child Left Behind federal law, which mandated that states establish tests for reading and math with at least 3 levels of scores: basic, proficient, and advanced (Colorado Department of Education, n.d.).  While much of NCLB has been amended since then, these tests are still used.  

But are such direct comparisons between school districts fair, or even enlightening?  Since US schools are primarily funded at the local level, not state or federal, there is a wide variety in school financial expenditure (Semuels, 2016).  Also, communities may have different levels of non-financial resources supporting education.  The test scores themselves are not directly comparable, since each state has a different set of tests.

To address these concerns, we performed an analysis using two sets of data: the Child Opportunity Index (Diversitydatakids.org, 2022) and the Stanford Educational Data Archive (Reardon et al., 2021).  The Child Opportunity Index (COI) is a holistic view of the resources available to children in a community, including indicators such as access to healthy food, 3rd grade reading and math scores, percentage of the population with health insurance, school financial expenditure, and average educational attainment by adults in the area.  The Stanford Educational Data Archive (SEDA) baselines state standardized test scores in reading and math against a common national test (the National Assessment of Educational Progress (NAEP)) in order to allow between-state comparisons.  In our analysis, we used the COI data to cluster school districts across the US and to predict SEDA scores.  This provided us with a view to which school districts are doing better than others from similar backgrounds.  
''')
#Report.subheader('dashboard')
Report.subheader('Methods')
Report.subheader('Data Cleaning')
Report.markdown(f'''We imputed missing values in COI and computed weighted averages of multiple census tracts before consolidating them by school district
''')
Report.subheader('Clustering')
Report.markdown(f'''We tried K-Means and DBSCAN clustering methods and found a better signal with K-Means.  Our goal was to use the cluster indicators as features in achievement score prediction
''')
Report.subheader('Prediction')

Report.subheader('Results')

Report.subheader('Discussion')
Report.markdown(f'''Learning from other states’ educational successes (ref EPI report)''')

Report.subheader('Citations')
Report.markdown(f'''Carnoy, M., García, E., & Khavenson, T. (2015, October 30). Bringing it back home:  Why state comparisons are more useful than international comparisons for improving U.S. education policy. Economic Policy Institute. https://www.epi.org/publication/bringing-it-back-home-why-state-comparisons-are-more-useful-than-international-comparisons-for-improving-u-s-education-policy/ 
Colorado Department of Education. (n.d.). Every Student Succeeds Act side-by-side. Retrieved October 27, 2022 from https://www.cde.state.co.us/fedprograms/nclbwaiveressasummary
Diversitydatakids.org. (2022). Child Opportunity Index (Version 2.0). [Data set]. https://data.diversitydatakids.org/dataset/coi20-child-opportunity-index-2-0-database?_external=True
Fahle, E. M., Chavez, B., Kalogrides, D., Shear, B. R., Reardon, S. F., & Ho, A. D. (2021). Stanford Education Data Archive: Technical Documentation (Version 4.1). http://purl.stanford.edu/db586ns4974
GreatSchools.org. (n.d.) GreatSchools ratings methodology report.  Retrieved November 6, 2022 from https://www.greatschools.org/gk/ratings-methodology
Klein, A. (2015, April 10). No Child Left Behind: An overview. Education Week. https://www.edweek.org/policy-politics/no-child-left-behind-an-overview/2015/04
National Center for Education Statistics. (2015). School district geographic relationship files. [Data set]. https://nces.ed.gov/programs/edge/Geographic/RelationshipFiles
Noelke, C., McArdle, N., Baek, M., Huntington, N., Huber, R., Hardy, E., & Acevedo-Garcia, D. (2020). Child Opportunity Index 2.0 Technical Documentation. http://diversitydatakids.org/research-library/research-brief/how-we-built-it
Reardon, S. F., Ho, A. D., Shear, B. R., Fahle, E. M., Kalogrides, D., Jang, H., & Chavez, B. (2021). Stanford Education Data Archive (Version 4.1). [Data set]. Stanford University. http://purl.stanford.edu/db586ns4974
Semuels, A. (2016, August 25). Good school, rich school; bad school, poor school: The inequality at the heart of America’s education system. The Atlantic. https://www.theatlantic.com/business/archive/2016/08/property-taxes-and-unequal-schools/497333/
United States Census Bureau. (2010). 2010: DEC redistricting data (PL 94-171). [Data set]. https://data.census.gov/cedsci/table?q=Decennial%20Census%20population&g=0100000US%241400000&d=DEC%20Redistricting%20Data%20%28PL%2094-171%29&tid=DECENNIALPL2020.P1

''')