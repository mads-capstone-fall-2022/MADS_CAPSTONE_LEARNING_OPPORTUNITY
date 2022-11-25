#import geopandas as gpd
import altair as alt
import pandas as pd
import streamlit as st
from gsheetsdb import connect
import pickle
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
Report, Dashboard, Test_Page = st.tabs(["Report Page", "Dashboard Page", "Test Page"])
Dashboard.title("Compare Achievment scores on the same Scale!")




# Create a connection object.
conn = connect()

# # Perform SQL query on the Google Sheet.
# # Uses st.cache to only rerun when the query changes or after 10 min.
# @st.cache(ttl=600)
# def run_query(query):
#     rows = conn.execute(query, headers=1)
#     rows = rows.fetchall()
#     return rows

# sheet_url = st.secrets["child_oppurtunity_input_file"] #st.secrets["seda_map_file"]
# rows = run_query(f'SELECT * FROM "{sheet_url}"')


# #print(type(rows))
# #seda_map_df  = pd.DataFrame(list(rows))
# child_opportunity_df = pd.DataFrame(list(rows))


# KA - try pickle instead of google sheets
with open('Data/coi_seda_display_1.pkl', 'rb') as f:
    coi_seda_1_df = pickle.load(f)

with open('Data/coi_seda_display_2.pkl', 'rb') as f:
    coi_seda_2_df = pickle.load(f)

child_opportunity_df = pd.concat([coi_seda_1_df, coi_seda_2_df])


with open('Data/feature_imp.pkl', 'rb') as f:
    feature_imp_df = pickle.load(f)


#add filters
v_segment = Dashboard.selectbox(
     'Which segment would you like to select',
     ('Segment 1', 'Segment 2', 'Segment 3','Segment 4'))

v_subject = Dashboard.selectbox(
     'Which subject would you like to select',
     ('Math', 'Reading'))

v_year_choice = Dashboard.slider(
    'Year:', min_value=2010, max_value=2015, step=1, value=2015)


#filter the dataframe
child_oppurtunity_df = child_opportunity_df[child_opportunity_df['seda_year'] == v_year_choice]
child_oppurtunity_df = child_opportunity_df[child_opportunity_df['cluster']==int(v_segment[-1])-1]
child_oppurtunity_df = child_opportunity_df[child_opportunity_df['subject']==v_subject]   
#st.set_page_config(layout="wide")
#fig = make_subplots(rows=1, cols=2)

# #fig = px.scatter_mapbox(seda_map_df, lat="latitude", lon="longitude", hover_name="NAME", hover_data=["GEOID"],
#                     #    color_discrete_sequence=["fuchsia"], zoom=3, height=300)
# fig = px.scatter_mapbox(child_opportunity_df,lat='latitude', lon='longitude', zoom=3, height=300, hover_name='sedaleaname',color_continuous_scale = 'rdylgn', color='subject', size='cs_mn_all_abs' ,text='sedaleaname')#, hover_name='subject')
# fig.update_layout(mapbox_style="open-street-map")
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.update_geos(fitbounds="locations")
# fig.update_layout(mapbox_bounds={"west": -180, "east": -50, "south": 20, "north": 90})
# Dashboard.plotly_chart(fig, use_container_width=True)
# #fig.show()
# #st.map(data=seda_map_df, zoom=None, use_container_width=True)
# #st.table(rows)

fig_bp_feat_imp = px.box(feature_imp_df, x='Variable', y='Importance', color='Cluster Name', height=600, width=1200)

Dashboard.plotly_chart(fig_bp_feat_imp)


child_opportunity_df['cluster'] = child_opportunity_df['cluster'].astype(str)

fig_sp_clusters = px.scatter(child_opportunity_df, 
                             x='Component 1', 
                             y='Component 2', 
                             color='cluster', 
                             category_orders={'cluster': ['0', '1', '2', '3']}, 
                             hover_name='NAME_LEA15',
                             log_x=True,
                             log_y=True,
                             width=800, 
                             height=600)



# Data loading for report sections
model_results_df = pd.read_csv('Data/model_results.csv')
cross_val_results_df = pd.read_csv('Data/cross_val_results.csv')


Report.header('Team Learning Opportunity Blog Post')
Report.text('Jay Korrapati and Katie Andrews')
Report.subheader('Introduction')

Report.markdown('''When school districts in the US are judged, it is usually by comparison to other districts.  Parents use ratings sites like GreatSchools - which uses test scores, graduation rates, and other data (GreatSchools.org, n.d.) - to compare schools when they are looking to move to a new area.  State governments use standardized test scores to rank schools and districts and identify struggling schools (Klein, 2015).  The standardized test scores used in both cases were designed at the state level in response to the 2001 No Child Left Behind federal law, which mandated that states establish tests for reading and math with at least 3 levels of scores: basic, proficient, and advanced (Colorado Department of Education, n.d.).  While much of NCLB has been amended since then, these tests are still used.  

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
Report.write(model_results_df)

Report.subheader('Results')

Report.subheader('Discussion')
Report.markdown(f'''Learning from other states' educational successes (ref EPI report)''')

Report.subheader('Citations')
Report.markdown('''<p style="padding-left: 2em; text-indent: -2em;">Carnoy, M., García, E., & Khavenson, T. (2015, October 30). <em>Bringing it back home:  Why state comparisons are more useful than international comparisons for improving U.S. education policy.</em> Economic Policy Institute. <a href="https://www.epi.org/publication/bringing-it-back-home-why-state-comparisons-are-more-useful-than-international-comparisons-for-improving-u-s-education-policy/">https://www.epi.org/publication/bringing-it-back-home-why-state-comparisons-are-more-useful-than-international-comparisons-for-improving-u-s-education-policy/</a> </p>
<p style="padding-left: 2em; text-indent: -2em;">Colorado Department of Education. (n.d.). <em>Every Student Succeeds Act side-by-side.</em> Retrieved October 27, 2022 from <a href="https://www.cde.state.co.us/fedprograms/nclbwaiveressasummary">https://www.cde.state.co.us/fedprograms/nclbwaiveressasummary</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Diversitydatakids.org. (2022). <em>Child Opportunity Index</em> (Version 2.0). [Data set]. <a href="https://data.diversitydatakids.org/dataset/coi20-child-opportunity-index-2-0-database?_external=True">https://data.diversitydatakids.org/dataset/coi20-child-opportunity-index-2-0-database?_external=True</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Fahle, E. M., Chavez, B., Kalogrides, D., Shear, B. R., Reardon, S. F., & Ho, A. D. (2021). <em>Stanford Education Data Archive: Technical Documentation (Version 4.1).</em> <a href="http://purl.stanford.edu/db586ns4974">http://purl.stanford.edu/db586ns4974</a></p>
<p style="padding-left: 2em; text-indent: -2em;">GreatSchools.org. (n.d.) <em>GreatSchools ratings methodology report.</em>  Retrieved November 6, 2022 from <a href="https://www.greatschools.org/gk/ratings-methodology">https://www.greatschools.org/gk/ratings-methodology</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Klein, A. (2015, April 10). <em>No Child Left Behind: An overview.</em> Education Week. <a href="https://www.edweek.org/policy-politics/no-child-left-behind-an-overview/2015/04">https://www.edweek.org/policy-politics/no-child-left-behind-an-overview/2015/04</a></p>
<p style="padding-left: 2em; text-indent: -2em;">National Center for Education Statistics. (2015). <em>School district geographic relationship files.</em> [Data set]. <a href="https://nces.ed.gov/programs/edge/Geographic/RelationshipFiles">https://nces.ed.gov/programs/edge/Geographic/RelationshipFiles</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Noelke, C., McArdle, N., Baek, M., Huntington, N., Huber, R., Hardy, E., & Acevedo-Garcia, D. (2020). <em>Child Opportunity Index 2.0 Technical Documentation.</em> <a href="http://diversitydatakids.org/research-library/research-brief/how-we-built-it">http://diversitydatakids.org/research-library/research-brief/how-we-built-it</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Reardon, S. F., Ho, A. D., Shear, B. R., Fahle, E. M., Kalogrides, D., Jang, H., & Chavez, B. (2021). <em>Stanford Education Data Archive</em> (Version 4.1). [Data set]. Stanford University. <a href="http://purl.stanford.edu/db586ns4974">http://purl.stanford.edu/db586ns4974</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Semuels, A. (2016, August 25). <em>Good school, rich school; bad school, poor school: The inequality at the heart of America's education system.</em> The Atlantic. <a href="https://www.theatlantic.com/business/archive/2016/08/property-taxes-and-unequal-schools/497333/">https://www.theatlantic.com/business/archive/2016/08/property-taxes-and-unequal-schools/497333/</a></p>
<p style="padding-left: 2em; text-indent: -2em;">United States Census Bureau. (2010). <em>2010: DEC redistricting data (PL 94-171).</em> [Data set]. <a href="https://data.census.gov/cedsci/table?q=Decennial%20Census%20population&g=0100000US%241400000&d=DEC%20Redistricting%20Data%20%28PL%2094-171%29&tid=DECENNIALPL2020.P1">https://data.census.gov/cedsci/table?q=Decennial%20Census%20population&g=0100000US%241400000&d=DEC%20Redistricting%20Data%20%28PL%2094-171%29&tid=DECENNIALPL2020.P1</a></p>
''', unsafe_allow_html=True)
