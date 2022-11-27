#import geopandas as gpd
#import altair as alt
import pandas as pd
import streamlit as st
#from gsheetsdb import connect
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import plotly.figure_factory as ff

st.set_page_config(layout="wide")
Report, Dashboard = st.tabs(["Report Page", "Dashboard Page"])




# # Create a connection object.
# conn = connect()

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


#### DATA LOADING ####
@st.cache(ttl=6000)
def load_data():
    with open('Data/coi_display.pkl', 'rb') as f:
        coi_df = pickle.load(f)

    with open('Data/seda_display.pkl', 'rb') as f:
        seda_df = pickle.load(f)

    with open('Data/feature_imp.pkl', 'rb') as f:
        feature_imp_df = pickle.load(f)

    with open('Data/school_data_for_map.pkl', 'rb') as f:
        school_data_for_map_df = pickle.load(f)

    model_results_df = pd.read_csv('Data/model_results.csv')
    cross_val_results_df = pd.read_csv('Data/cross_val_results.csv')

    return coi_df, seda_df, feature_imp_df, model_results_df, cross_val_results_df, school_data_for_map_df

coi_df, seda_df, feature_imp_df, model_results_df, cross_val_results_df, school_data_for_map_df = load_data()

cluster_df = coi_df.copy()


#### DASHBOARD SECTION ####
Dashboard.title('Compare Achievment Scores on the Same Scale')


# add filters at the top
filter_col1,filter_col2, filter_col3 = Dashboard.columns(3)
v_segment = filter_col1.selectbox(
     'Which cluster would you like to select',
     ('All Clusters', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'))

v_subject = filter_col2.selectbox(
     'Which subject would you like to select',
     ('Math', 'Reading'))

v_year_choice =  filter_col3.radio(
    "Select a Year",
    (2016, 2017, 2018))

#filter_col1.slider(
   # 'Year:', min_value=2016, max_value=2018, step=1, value=2016)


#join the geo code info to seda data for map visual
#get a subset of the dataframe columns for building the dashboard
seda_df = seda_df[['LEAID','NAME_LEA15','fips','stateabb','sedalea','sedaleaname','subject','grade','seda_year','cs_mn_all','Cluster Name']]
#merge the two data sets so the school district data and latitude and longitude data are in the same place
seda_df = seda_df.merge(school_data_for_map_df, on ='sedalea')
#clean up the resulting dataframe columns to include on the fields We need for dashboard
seda_df  = seda_df[['NAME_LEA15', 'stateabb', 'sedalea', 
       'subject', 'grade', 'seda_year', 'cs_mn_all', 'Cluster Name',
       'latitude', 'longitude']]
#rename the sedalean name column
seda_df = seda_df.rename(columns={"NAME_LEA15": "sedalea_name"})
#convert the data types for the fields latitude(float) , longitude(float), and year(year)
seda_df.loc[:,'latitude'] = seda_df['latitude'].astype(str).astype(float)
seda_df.loc[:,'longitude'] = seda_df['longitude'].astype(str).astype(float)
seda_df.loc['seda_year'] = pd.to_datetime(seda_df.loc[:,'seda_year'], format='%Y')








# filter the dataframes
if v_segment == 'All Clusters':
    seda_disp_df = seda_df.copy()
    seda_disp_df = seda_disp_df[(seda_disp_df['seda_year']==v_year_choice) & (seda_disp_df['subject']==v_subject)]

    feature_imp_disp_df = feature_imp_df.copy()

else:
    seda_disp_df = seda_df[(seda_df['Cluster Name']==v_segment) & (seda_df['seda_year']==v_year_choice) & (seda_df['subject']==v_subject)]
    
    feature_imp_disp_df = feature_imp_df[feature_imp_df['Cluster Name']==v_segment]

map_visual_col , dist_plot_visual = Dashboard.columns(2)
#add data processing steps for the map visual

seda_disp_df['sign'] =  np.where(seda_disp_df['cs_mn_all'] >= 0, 'Positive', 'Negative')
seda_disp_df['cs_mn_all_abs'] = np.abs(seda_disp_df['cs_mn_all'])
seda_disp_df.loc[:,'cs_mn_all_abs'] = seda_disp_df['cs_mn_all_abs'].astype(str).astype(float)

#create the map configuration

fig_map = px.scatter_mapbox(data_frame=seda_disp_df,lat='latitude', lon='longitude', color='sign',color_discrete_sequence=px.colors.qualitative.G10,
                        zoom = 3,size='cs_mn_all_abs' ,text='sedalea_name', color_discrete_map = {'Negative': '#AB63FA', 'Positive':'#FECB52'},hover_data = ['sedalea_name','stateabb'])#, hover_name='subject')
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig = px.scatter_mapbox(data_frame=child_oppurtunity_index_data_df, lat='latitude', lon='longitude', color='sign', text='sedaleaname', hover_name='subject', size='cs_mn_all_abs')

fig_map.update_layout(mapbox_style="open-street-map", autosize=True)
map_visual_col.plotly_chart(fig_map, use_container_width=True)


#Display the distribution plot for all the clusters
# Add histogram data
x1 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 1')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')
x2 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 2')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')
x3 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 3')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')
x4 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 4')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')

# Group data together
hist_data = [x1, x2, x3, x4]
group_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
# Create distplot with custom bin_size
fig_dist = ff.create_distplot(
        hist_data, group_labels)
fig_dist.update_layout(autosize=True)
# Plot
dist_plot_visual.plotly_chart(fig_dist, use_container_width=True)









#get summary stats
#get the count of distinct states the school districts are in
v_distinct_states  = seda_disp_df.stateabb.nunique()
#get the count of school districts
v_distinct_school_districts = seda_disp_df.sedalea.nunique()
#count of negative scores
v_negative_score_count = seda_disp_df[seda_disp_df['cs_mn_all'].lt(0)]['cs_mn_all'].count()
#count of scores greater than or equal to zero
v_positive_score_count = seda_disp_df[seda_disp_df['cs_mn_all'].ge(0)]['cs_mn_all'].count()

col1, col2, col3, col4 = Dashboard.columns(4)
col1.metric(label="Number of States", value= v_distinct_states )
col2.metric(label="Number of School Districts", value=v_distinct_school_districts)
col3.metric(label="Number of Negative Scores", value=v_negative_score_count)
col4.metric(label="Number of Positive Scores", value=v_positive_score_count)


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

fig_bp_feat_imp = px.box(feature_imp_disp_df, 
                         x='Variable', 
                         y='Importance', 
                         color='Cluster Name', 
                         height=600, 
                         width=1200, 
                         title='Test Set Model Feature Importance')

Dashboard.plotly_chart(fig_bp_feat_imp)




#### REPORT SECTION ####

Report.title('Evaluating US School District Achievement Scores Based on Community Resource Levels')
Report.markdown('Team Learning Opportunity: Jay Korrapati and Katie Andrews')
Report.header('Introduction', anchor='introduction')

Report.markdown('''When school districts in the US are judged, it is usually by comparison to other districts.  Parents use ratings sites like GreatSchools - which uses test scores, graduation rates, and other data (GreatSchools.org, n.d.) - to compare schools when they are looking to move to a new area.  State governments use standardized test scores to rank schools and districts and identify struggling schools (Klein, 2015).  The standardized test scores used in both cases were designed at the state level in response to the 2001 No Child Left Behind federal law, which mandated that states establish tests for reading and math with at least 3 levels of scores: basic, proficient, and advanced (Colorado Department of Education, n.d.).  While much of NCLB has been amended since then, these tests are still used.  

But are such direct comparisons between school districts fair, or even enlightening?  Since US schools are primarily funded at the local level, not state or federal, there is a wide variety in school financial expenditure (Semuels, 2016).  Also, communities may have different levels of non-financial resources supporting education.  The test scores themselves are not directly comparable, since each state has a different set of tests.

To address these concerns, we performed an analysis using two sets of data: the Child Opportunity Index (Diversitydatakids.org, 2022) and the Stanford Educational Data Archive (Reardon et al., 2021).  The Child Opportunity Index (COI) is a holistic view of the resources available to children in a community, including indicators such as access to healthy food, 3rd grade reading and math scores, percentage of the population with health insurance, school financial expenditure, and average educational attainment by adults in the area.  The Stanford Educational Data Archive (SEDA) baselines state standardized test scores in reading and math against a common national test (the National Assessment of Educational Progress (NAEP)) in order to allow between-state comparisons.  In our analysis, we used the COI data to cluster school districts across the US and to predict SEDA scores.  This provided us with a view to which school districts are doing better than others from similar backgrounds.  
''')
#Report.subheader('dashboard')
Report.header('Methods', anchor='methods')
Report.subheader('Data Cleaning')
Report.markdown(f'''We imputed missing values in COI and computed weighted averages of multiple census tracts before consolidating them by school district
''')
Report.subheader('Clustering')
Report.markdown(f'''We tried K-Means and DBSCAN clustering methods and found a better signal with K-Means.  Our goal was to use the cluster indicators as features in achievement score prediction
''')
Report.subheader('Prediction')
Report.write(cross_val_results_df)

Report.header('Results', anchor='results')
Report.subheader('Clustering Results')

fig_sp_clusters = px.scatter(cluster_df, 
                             x='Component 1', 
                             y='Component 2', 
                             color='Cluster Name', 
                             #category_orders={'cluster': ['0', '1', '2', '3']}, 
                             hover_name='NAME_LEA15',
                             log_x=True,
                             log_y=True,
                             width=800, 
                             height=600,
                             title='School District Clusters from COI Indicators'
                             )
fig_sp_clusters.update_xaxes(showgrid=False)
fig_sp_clusters.update_yaxes(showgrid=False)

Report.plotly_chart(fig_sp_clusters)

Report.subheader('Prediction Results')
Report.write(model_results_df)

Report.header('Discussion', anchor='discussion')
Report.markdown(f'''Learning from other states' educational successes (ref EPI report)''')

Report.header('References', anchor='references')
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

Report.header('Statement of Work', anchor='statement_of_work')

Report.header('Appendix', anchor='appendix')
