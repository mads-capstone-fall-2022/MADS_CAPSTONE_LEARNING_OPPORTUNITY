#import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio

st.set_page_config(layout="wide")
Report, Dashboard = st.tabs(["Report Page", "Dashboard Page"])

pio.templates['TLO'] = go.layout.Template(
    layout=go.Layout(font=dict(family='Rockwell', size=16), title_font=dict(size=24)
    )
)
pio.templates.default = 'plotly+TLO'



#### DATA LOADING ####
@st.cache(ttl=6000)
def load_data():
    with open('Data/coi_display.pkl', 'rb') as f:
        coi_df = pickle.load(f)

    with open('Data/seda_display.pkl', 'rb') as f:
        seda_df = pickle.load(f)

    with open('Data/feature_imp.pkl', 'rb') as f:
        feature_imp_df = pickle.load(f)

    with open('Data/clusters.pkl', 'rb') as f:
        cluster_df = pickle.load(f)
    
    model_results_df = pd.read_csv('Data/model_results.csv')
    cross_val_results_df = pd.read_csv('Data/cross_val_results.csv')

    return coi_df, seda_df, feature_imp_df, cluster_df, model_results_df, cross_val_results_df

coi_df, seda_df, feature_imp_df, cluster_df, model_results_df, cross_val_results_df = load_data()



#### DASHBOARD SECTION ####
Dashboard.title('Student Achievment Scores on the Same Scale')


# add filters at the top
filter_col1,filter_col2, filter_col3 = Dashboard.columns(3)
v_segment = filter_col1.selectbox(
     'Which cluster would you like to select',
     ('All Clusters', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'))

st.text(" ")
st.text(" ")

v_subject = filter_col2.selectbox(
     'Which subject would you like to select',
     ('Math', 'Reading'))

st.text(" ")
st.text(" ")

v_year_choice =  filter_col3.radio(
    "Select a Year",
    (2016, 2017, 2018), horizontal =True)

#filter_col1.slider(
   # 'Year:', min_value=2016, max_value=2018, step=1, value=2016)


#rename the sedalean name column
seda_df = seda_df.rename(columns={"NAME_LEA15": "sedalea_name"})
#convert the data types for the field year(year)
# seda_df.loc[:, 'seda_year'] = pd.to_datetime(seda_df.loc[:, 'seda_year'], format='%Y')
# print(seda_df[seda_df['seda_year'].dt.strftime('%Y')=='2016'].shape)


#added some space after the first line of controls on the streamlit app screen
#st.markdown('##')
st.text(" ")
st.text(" ")




# filter the dataframes
if v_segment == 'All Clusters':
    seda_disp_df = seda_df.iloc[:, :]
    seda_disp_df = seda_disp_df[(seda_disp_df['seda_year']==v_year_choice) & (seda_disp_df['subject']==v_subject)]

    feature_imp_disp_df = feature_imp_df.iloc[:, :]

else:
    seda_disp_df = seda_df[(seda_df['Cluster Name']==v_segment) & (seda_df['seda_year']==v_year_choice) & (seda_df['subject']==v_subject)]
    
    feature_imp_disp_df = feature_imp_df[feature_imp_df['Cluster Name']==v_segment]

map_visual_col , dist_plot_visual = Dashboard.columns(2)
#add data processing steps for the map visual

seda_disp_df['sign'] =  np.where(seda_disp_df['cs_mn_all'] >= 0, 'Positive', 'Negative')
seda_disp_df['cs_mn_all_abs'] = np.abs(seda_disp_df['cs_mn_all'])


#create the map configuration
fig_map = px.scatter_mapbox(data_frame=seda_disp_df,lat='latitude', lon='longitude', color='sign',color_discrete_sequence=px.colors.qualitative.G10,
                        zoom = 2,size='cs_mn_all_abs' ,title='Relative mean 4th grade scores across the USA', text='sedalea_name', color_discrete_map = {'Negative': '#AB63FA', 'Positive':'#FECB52'})
#fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig = px.scatter_mapbox(data_frame=child_oppurtunity_index_data_df, lat='latitude', lon='longitude', color='sign', text='sedaleaname', hover_name='subject', size='cs_mn_all_abs')

fig_map.update_layout(mapbox_style="open-street-map", autosize=True)
map_visual_col.plotly_chart(fig_map, use_container_width=True)


#Display the distribution plot for all the clusters
# Add histogram data
#x1 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 1')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')
#x2 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 2')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')
#x3 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 3')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')
#x4 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 4')&(seda_df['seda_year']==2016)&(seda_df['subject']=='Math')]['cs_mn_all'], dtype='float')

#make the histogram selection dynamic
x1 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 1')&(seda_df['seda_year']==v_year_choice)&(seda_df['subject']==v_subject)]['cs_mn_all'], dtype='float')
x2 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 2')&(seda_df['seda_year']==v_year_choice)&(seda_df['subject']==v_subject)]['cs_mn_all'], dtype='float')
x3 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 3')&(seda_df['seda_year']==v_year_choice)&(seda_df['subject']==v_subject)]['cs_mn_all'], dtype='float')
x4 = np.array(seda_df[(seda_df['Cluster Name']=='Cluster 4')&(seda_df['seda_year']==v_year_choice)&(seda_df['subject']==v_subject)]['cs_mn_all'], dtype='float')



# Group data together
hist_data = [x1, x2, x3, x4]
group_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
# Create distplot with custom bin_size
fig_dist = ff.create_distplot(
        hist_data, group_labels)

fig_dist.update_layout(autosize=True, title="Distribution Plot-Selected Cluster Relative to Others")
# Plot
dist_plot_visual.plotly_chart(fig_dist, use_container_width=True)



# Prep for COI histograms on report page
hist_coi_cols = ['ED_SCHPOV', 'ED_MATH', 'ED_READING', 'ED_ATTAIN', 'SE_SINGLE', 'HE_HLTHINS', 'HE_PM25', 'HE_RSEI']
hist_coi_names = ['School Poverty', '3rd Grade Math Proficiency', '3rd Grade Reading Proficiency', 'Adult Ed Attainment', 
                  'Single-Headed Households', 'Health Insurance Coverage', 'Airborne Microparticles', 'Industrial Pollutants']
hist_coi_labels = {hist_coi_cols[i]: hist_coi_names[i] for i in range(len(hist_coi_names))}

hist_coi = coi_df.melt(id_vars=['LEAID', 'NAME_LEA15', 'Cluster Name'], value_vars=hist_coi_cols, var_name='COI Variable', value_name='Value').reset_index()
hist_coi['COI Variable'] = hist_coi['COI Variable'].replace(hist_coi_labels)

coi_hist_1 = hist_coi[hist_coi['COI Variable'].isin(hist_coi_names[:4])]
coi_hist_2 = hist_coi[hist_coi['COI Variable'].isin(hist_coi_names[4:])]



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
#compute the percentage of schools with positive and negative scores
v_negative_score_percentage = round((v_negative_score_count / (v_negative_score_count + v_positive_score_count))*100 , 2)
v_positive_score_percentage = round((v_positive_score_count / (v_negative_score_count + v_positive_score_count))*100 , 2)


#display the controls
col1.metric(label="Number of States", value= v_distinct_states , help="The number of States the School districts in the above selection fall in")
col2.metric(label="Number of School Districts", value=v_distinct_school_districts, help="Number of School districts in the filter selection above")
col3.metric(label="Number of Negative Scores", value=v_negative_score_count, delta='-'+str(v_negative_score_percentage)+'%', help="Number of School districts with negative scores relative to the mean 4th grade score")
col4.metric(label="Number of Positive Scores", value=v_positive_score_count, delta=str(v_positive_score_percentage)+'%', help="Number of School districts with positive scores relative to the mean 4th grade score")


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
                        # height=800, 
                        # width=1200, 
                         title='Model Feature Importance')
fig_bp_feat_imp.update_layout( autosize=True)
Dashboard.plotly_chart(fig_bp_feat_imp, use_container_width=True)




#### REPORT SECTION ####

Report.title('Evaluating US School District Achievement Scores Based on Community Resource Levels')
Report.markdown('Team Learning Opportunity: Jayachandra Korrapati and Katie Andrews')


Report.header('Introduction', anchor='introduction')
Report.markdown('''When school districts in the US are judged, it is usually by comparison to other districts.  Parents use ratings sites like GreatSchools - which uses test scores, graduation rates, and other data (GreatSchools.org, n.d.) - to compare schools when they are looking to move to a new area.  State governments use standardized test scores to rank schools and districts and identify struggling schools (Klein, 2015).  The standardized test scores used in both cases were designed at the state level in response to the 2001 No Child Left Behind federal law, which mandated that states establish tests for reading and math with at least 3 levels of scores: basic, proficient, and advanced (Colorado Department of Education, n.d.).  While much of NCLB has been amended since then, these tests are still used.  
''')
Report.markdown('''But are such direct comparisons between school districts fair, or even enlightening?  Since US schools are primarily funded at the local level, not state or federal, there is a wide variety in school financial expenditure (Semuels, 2016).  Also, communities may have different levels of non-financial resources supporting education.  The test scores themselves are not directly comparable, since each state has a different set of tests.
''')
Report.markdown('''To address these concerns, we performed an analysis using two sets of data: the Child Opportunity Index (Diversitydatakids.org, 2022) and the Stanford Educational Data Archive (Reardon et al., 2021). The Child Opportunity Index (COI) is a holistic view of the resources available to children in a community, including indicators such as access to healthy food, 3rd grade reading and math scores, percentage of the population with health insurance, school financial expenditure, and average educational attainment by adults in the area. The Stanford Educational Data Archive (SEDA) baselines state standardized test scores in reading and math against a common national test (the National Assessment of Educational Progress (NAEP)) in order to allow between-state comparisons. For SEDA scores, 0 represents the national average, with positive scores representing grade years above average and negative scores representing grade years below average.  A score of 1 means that the students in the district scored one grade level higher than average; -1 means that they scored one grade level lower than average.
''')
Report.markdown('''In our analysis, we used the COI data to cluster school districts across the US and to predict SEDA scores. This provided us with a view to which school districts are doing better than others from similar backgrounds.  With the results of this analysis, we created a dashboard to allow parents and school administrators to explore the COI and SEDA data.  This dashboard is available on the second tab at the top of this page.
''')


Report.header('Methods', anchor='methods')
Report.subheader('Data Cleaning')
Report.markdown('''The biggest challenge in preparing our two main datasets - Child Opportunity Index (COI) and Stanford Educational Data Archive (SEDA) - was getting the two on the same geographic scale.  The COI data is at the census tract level.   The SEDA data we chose to use is at the school district level.  There can be multiple census tracts in a district and a tract may also be in multiple districts.  To resolve this, we created a scaled population value for each tract/school district combination by  multiplying the total population of the tract by the percentage of the tract's land area that is in the school district.  This factor was then used as the weight in a weighted average for each COI indicator for each school district.
''')
Report.latex(r'''COIindicator_{weighted} = \frac {\sum_1^n {COIindicator * (TractPop * LandAreaPercent)}} {\sum_1^n {TractPop * LandAreaPercent}}''')
Report.markdown('''Before computing the weighted average, we performed a train/test split on our data based on the Local Educational Authority Identifiers (LEAIDs).  We then imputed missing values in the COI indicators.  If we had not done so, the averaging process would essentially have treated missing values as zeros.  Since these values were not yet centered at zero, this could represent a high, low, or even outlier value in the indicator's distribution.  For a more consistent impact, we used the column median to fill missing values.  Once imputed and averaged, the indicators were scaled to a mean of zero and standard deviation of 1.  Both the fitted imputer and the fitted scaler were saved for later use when the test set was process similarly.
''')
Report.markdown('''We chose to use the most current available data, so we used the 2015 COI data (a 2010 set is also available).  This dataset includes as features the 3rd grade reading and math scores sourced from SEDA.  Therefore, when preparing the data for prediction tasks, we removed the 3rd grade scores from the target SEDA set and predicted only grades 4 through 8.  Since we were starting from 2015 COI data, we also subsetted the SEDA data to include only 2016-2018 scores.  The scores represented on the dashboard are only for the 4th grade for each of the three years.  The SEDA project directly measured 4th grade scores - that is one of the years that the NAEP test is taken 0 - so these scores are not interpolated as some of the other grade years are.  
''')


Report.subheader('Clustering')
Report.markdown('''The goal of our clustering methods was to use the results to separate school districts into groups with similar patterns of resource availability.  These groups would allow us to compare SEDA achievement scores between similar districts.
''')
Report.markdown('''We first performed Principle Component Analysis on the COI data, then clustered based on the first 11 components.  We chose 11 because those components represented explained 80% of the variance in the data.
''')
Report.markdown('''# Why PCA?
''')
Report.markdown('''We tried two clustering methods: K-Means and DBSCAN.  DBSCAN is a density-based clustering algorithm and it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions. In our case the DBSCAN method found one cluster and a lot of noise points. Our hypothesis is that most of the data points in computed PCA space (reduced dimensions in the latent space) are close to each other.  K-Means was able to separate this dense space to arrive at better groupings.
''')


Report.subheader('Prediction')
Report.markdown('''Our predictor variables were the COI indicators; population levels (both children-only and total population); the SEDA year, grade level, and subject (reading or math); and the cluster label resulting from the K-Means model.  We chose the mean score for all students as the target variable from the SEDA dataset for our prediction tasks.  Using this target, we tried several prediction methods, outlined in the table below, including linear, tree-based, and histogram-based models.  Once the most promising model was selected, we performed hyperparameter tuning cross-validation to optimize model performance. 
''')
Report.markdown('''Our goal in creating these models was less to create a highly accurate prediction and more to investigate the predictive power of the community resource indicators from the COI dataset.  We explored running a single model for all school districts and also running different predictive models for each cluster. The intuition was that grade achievement scores are related to the COI data, so segmenting on clustered school districts would give us better prediction accuracy and insight into potentially different COI variables impacting each model.
''')


Report.header('Results', anchor='results')
Report.subheader('Clustering Results')
Report.markdown('''As an unsupervised learning method, K-Means clustering does not have a ground truth on which to base an accuracy score.  Also, K-Means requires a pre-specified number of clusters as a parameter.  We chose cluster inertia - a measure of closeness within the cluster and difference between clusters - to determine the best number. We used a scree plot to identify 4 clusters as optimal based on the change in cluster inertia as the number of clusters  increased.
''')
Report.markdown('''We discovered that one of the resulting clusters was very small (4 districts), but consisted of 4 enormous metropolitan districts (New York, Los Angeles, Dade County (Miami), and Chicago). The other 3 clusters were both more diverse and closer in size with 4710, 2539, and 3594 school districts, respectively.  The below plot shows the clusters based on the first 2 PCA components.  As can be seen, the small cluster (Cluster 2), is in the upper right of the plot.  The remaining clusters are densely packed.  This density is what caused the DBSCAN clustering algorithm to be ineffective, just finding 1 cluster and many outliers.  This visualization uses log-scale axes to spread out the tightly grouped clusters to make the divisions more visible. 
''')


fig_sp_clusters = px.scatter(cluster_df, 
                             x='Component 1', 
                             y='Component 2', 
                             color='Cluster Name', 
                             category_orders={'Cluster Name': ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster  4']}, 
                             hover_name='NAME_LEA15',
                             log_x=True,
                             log_y=True,
                             width=800, 
                             height=600,
                             title='School District Clusters from COI Indicators'
                             )
fig_sp_clusters.update_xaxes(showgrid=False)
fig_sp_clusters.update_yaxes(showgrid=False)

Report.plotly_chart(fig_sp_clusters, sharing='streamlit')

Report.markdown('''There is a caveat about our clustering results.  When the trained K-Means cluster model was applied to the held-out test set, we found that no new school districts were added to Cluster 3 (the large, metropolitan districts cluster).
''')
Report.markdown('''We created dashboard visualizations to represent these clusters of school districts across the US and represent the relative achievement scores. This allows dashboard users to more objectively compare achievement across school districts. 
''')
Report.markdown('''# A year over year comparison would help track how scores are changing.  Are we doing this?
''')


Report.subheader('Prediction Results')
Report.markdown('''The primary goal of our prediction activities was to identify important features in the COI data - both at the nation level and across different clusters of school districts.  To accomplish this, we applied a variety of modeling methods from the scikit-learn package (Pedregosa et al., 2011).  The results are summarized in the following table.  We used the $R^2$ (coefficient of determination) scoring method to evaluate the models.  The best score for this method is 1, with 0 representing a constant prediction of the average target value and negative scores being indefinitely worse.
''')

fig_model_results = go.Figure(data=[go.Table(columnwidth = [300, 300, 100, 100, 100],
                                             header=dict(values=list(model_results_df.columns),
                                                         fill_color='black', 
                                                         font=dict(color='white', size=16)), 
                                             cells=dict(values=[model_results_df['Model'], model_results_df['Hyperparameters'], 
                                                                model_results_df['Cluster'], model_results_df['Training Set Score'], 
                                                                model_results_df['Test Set Score']], 
                                                        align=['left', 'left', 'left', 'right', 'right'],
                                                        fill_color='grey', 
                                                        line_color='white',
                                                        font=dict(color='white', size=14),
                                                        format=[None, None, None, '.4f', '.4f']))])
fig_model_results.update_layout(
    autosize=False,
    margin_b=0,
    height=400,
    width=1000,
    showlegend=False,
    title_text='Predictive Model Results',
)

Report.plotly_chart(fig_model_results, sharing='streamlit')

Report.markdown('''Overall, the histogram-based gradient boosting tree regression method produced a similar score on the training set to the more common, non-histogram gradient boosting regression, but performed much better, efficiency-wise, on over 150,000 rows of training data.  This better performance allowed us to do a grid search with cross-validation of learning rate and maximum depth parameters, leading to an improved score on random subsets of the training data.  Cross-validation result details are in the Appendix.
''')
Report.markdown('''Once we decided on the histogram-based gradient boosting tree regressor method, we trained models both on all the training data at once, and also as separate models for each cluster.  Grid search resulted in different hyperparameters for the cluster models.  
The clusters also had widely varying $R^2$ scores on the training data and the test set.  The overall model was moderately successful and had correct looking residuals (plot in Appendix), but the cluster 1 performed much worse.  
''')
Report.markdown('''To further investigate our model performance, we looked at feature importance for each model and examined the cluster characteristics for those features.  We chose to use permutation feature importance, which is a method that rotates through features, removing each in turn from the model.  The calculated importance is the decrease in the $R^2$ score with the feature removed.  The below interactive box plot shows the top 5 features for each model.
''')
Report.markdown('''An outlier in this plot is Cluster 3's top feature, child population.  Cluster 3 is the large-district cluster, which only consists of 3 districts once linked to the SEDA data, because the state of New York did not submit scores for 2016-2018.  Removing that outlier by clicking the legend item for Cluster 3 makes the rest of the importances more easily visible.
''')


fig_rt_feat_imp = px.box(feature_imp_df, 
                         x='Variable', 
                         y='Importance', 
                         color='Cluster Name', 
                         height=800, 
                         width=1200, 
                         title='Model Feature Importance')

Report.plotly_chart(fig_rt_feat_imp, use_container_width=True, sharing='streamlit')


Report.markdown('''Most of the features were from the COI indicators, with the exception of the SEDA variables **Grade**, **School Year**, **Subject (Read/Math)** and the calculated **Cluster ID**.  There was substantial overlap in the feature lists between models.  We expected the **3rd Grade Reading** and **3rd Grade Math** features to be important, since it makes intuitive sense that 3rd grade scores would impact later grade level scores.  The importance of these features helps show that the models reflect real impactful characteristics.  The **School Poverty** feature in particular was highly important in 4 out of 5 models.  This aligns with other studies of school performance that find that poverty levels are greatly predictive of success relative to other schools (Semuels, 2016).
''')
Report.markdown('''The below set of histograms shows the distributions of the top 8 COI factors.  In many of these plots, Cluster 1 is roughly in the center, with Cluster 2 and Cluster 4 tending toward opposite sides.  For example, on the **School Poverty** indicator, Cluster 2's distribution peaks in the negative numbers, versus Cluster 4's peak in the positive numbers.  This means that more districts in Cluster 2 have low poverty levels, where Cluster 4 has more districts with higher poverty levels.  From this, and based on the importance of this factor in the overall model, we expect that there will be noticeable differences in the scores between these two clusters.  Exploring from the dashboard, it can be seen that Cluster 2 has more scores below the national average and Cluster 4 has more scores above the national average.  Cluster 1 shows a middle level of poor districts and also a balanced number of above-average and below-average scores.   From this, it appears that lower poverty levels correlate with higher scores and higher poverty levels with lower scores.
''')

sp_coi_hist_1 = px.histogram(coi_hist_1, 
                             x='Value', 
                             color='Cluster Name', 
                             category_orders={'Cluster Name': ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster  4']}, 
                             facet_col='COI Variable', 
                             marginal='violin',
                             nbins=100,
                             width=1200,
                             height=500,
                             title='Important COI Feature Distributions')
sp_coi_hist_2 = px.histogram(coi_hist_2, 
                             x='Value', 
                             color='Cluster Name', 
                             category_orders={'Cluster Name': ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster  4']}, 
                             facet_col='COI Variable', 
                             marginal='violin',
                             nbins=100,
                             width=1200,
                             height=500)

with st.container():
    Report.plotly_chart(sp_coi_hist_1, use_container_width=True, sharing='streamlit')
    Report.plotly_chart(sp_coi_hist_2, use_container_width=True, sharing='streamlit')


Report.markdown('''The features in the above distributions can be divided into two groups: positive factors and negative factors.  The pattern between clusters 1, 2, and 4 are consistent.  For positive factors - 3rd grade scores, health insurance coverage - Cluster 2 (higher scoring) shows higher values and Cluster 4 (lower scores) shows lower values.  For negative factors - poverty, pollution - the opposite is true.  In each case, Cluster 1 is roughly in the middle.
''')


Report.header('Discussion', anchor='discussion')
Report.markdown('''A report from the Economic Policy Institute argued that the US can best learn how to improve public schools by looking to other states that are more successful, rather than by comparing US schools with other countries (Carnoy & Khavenson, 2015).  Their argument was that other industrialized countries can have very different qualities from the US, such as nationally-run schools, instead of state-run, or more homogenous populations, compared to the diversity of the US population.  However, comparing achievement levels across states is complicated, due to differing tests.  In addition, comparing at the state level makes it difficult to get detailed information, since there are many school districts with varying performance in each state.  Local differences in school funding and other characteristics can be substantial (Semuels, 2016).  Therefore, to really get detailed, informative comparisons, we believe that the school district level is best. 
''')
Report.markdown('''We chose the Child Opportunity Index (COI) as the basis for our school district clusters and our predictive models because its factors provide a holistic view of a community's resources for children, not simply economic factors.  We found that, while poverty was a highly important feature in our models, other non-economic features - such as pollution - were also impactful.  We chose the Stanford Educational Data Archive (SEDA) mean reading and math scores as our target variable for prediction because we wanted to be able to compare school districts across state lines, which is difficult due to different tests being used in each state.  The Stanford project baselined those state-specific scores into ones on the same, national scale.
''')
Report.markdown('''Our goals for our analysis of these datasets were:
- To identify a small number of school district clusters with distinct characteristics based on the Child Opportunity Index data
- To identify important features in models predicting the school district's mean achievement score based on the COI data for all clusters and also individual clusters
''')
Report.markdown('''In our analysis, we found 4 clusters.  One of them was essentially a cluster of outliers - extremely large urban school districts.  The other three clusters divided the remaining districts roughly in thirds.   When we examined permutation feature importance for histogram-based gradient boosting tree regression models trained either on all clusters together or on individual clusters, we found correlations between the distributions of important features and the amount of districts reporting above- or below-average scores.  For the **School Poverty** feature, the pattern was especially clear, with higher levels of poverty correlating to more districts reporting low mean scores. 
''')
Report.markdown('''The dashboard we created based on this analysis can be used by parents and school administrators to identify districts that are doing well or poorly compared to others with other districts that have similar characteristics.  This information can be used to support improvements in school achievement by providing fairer comparisons between districts. 
''')
Report.markdown('''This project had several limitations, one of which was due to an ethical concern with the SEDA data.  While individual school-level data is available from SEDA, some of the values are obscured due to privacy concerns, particularly if the school has a small population of an ethnic minority children.  To avoid this issue, we used data from the district level, which was not obscured.  Another limitation was choosing to predict just one of the many possible outcome variables in the SEDA data (overall mean score).
''')
Report.markdown('''An interesting opportunity for future work is to include US Census data on gender, race, and ethnicity and use those factors in concert with the COI data to predict scores for sub-groups and gaps between sub-group scores.  SEDA calculated gaps, for example, for girls versus boys and white versus hispanic sub-populations.  Additionally, SEDA has just released new data for 2019-2021.  This data could be explored to understand COVID impacts on achievement scores by using the 2015 COI data and 2016-2018 SEDA data to create time-series predictions of scores and comparing them with the newly-released actual scores for 2019-21.
''')


Report.header('Statement of Work', anchor='statement_of_work')
Report.markdown('''Katie Andrews: Data processing, prediction methods and feature importance, and report''')
Report.markdown('''Jayachandra Korrapati: Data processing, Principal Component Analysis, clustering methods, environment setup and configuration, and dashboard''')


Report.header('References', anchor='references')
Report.markdown('''
<p style="padding-left: 2em; text-indent: -2em;">Carnoy, M., García, E., & Khavenson, T. (2015, October 30). Bringing it back home:  Why state comparisons are more useful than international comparisons for improving U.S. education policy. Economic Policy Institute. <a href="https://www.epi.org/publication/bringing-it-back-home-why-state-comparisons-are-more-useful-than-international-comparisons-for-improving-u-s-education-policy/">https://www.epi.org/publication/bringing-it-back-home-why-state-comparisons-are-more-useful-than-international-comparisons-for-improving-u-s-education-policy/</a> </p>
<p style="padding-left: 2em; text-indent: -2em;">Colorado Department of Education. (n.d.). <em>Every Student Succeeds Act side-by-side.</em> Retrieved October 27, 2022 from <a href="https://www.cde.state.co.us/fedprograms/nclbwaiveressasummary">https://www.cde.state.co.us/fedprograms/nclbwaiveressasummary</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Diversitydatakids.org. (2022). <em>Child Opportunity Index</em> (Version 2.0). [Data set]. <a href="https://data.diversitydatakids.org/dataset/coi20-child-opportunity-index-2-0-database?_external=True">https://data.diversitydatakids.org/dataset/coi20-child-opportunity-index-2-0-database?_external=True</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Fahle, E. M., Chavez, B., Kalogrides, D., Shear, B. R., Reardon, S. F., & Ho, A. D. (2021). <em>Stanford Education Data Archive: Technical Documentation (Version 4.1).</em> <a href="http://purl.stanford.edu/db586ns4974">http://purl.stanford.edu/db586ns4974</a></p>
<p style="padding-left: 2em; text-indent: -2em;">GreatSchools.org. (n.d.) <em>GreatSchools ratings methodology report.</em>  Retrieved November 6, 2022 from <a href="https://www.greatschools.org/gk/ratings-methodology">https://www.greatschools.org/gk/ratings-methodology</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Klein, A. (2015, April 10). No Child Left Behind: An overview. <em>Education Week.</em> <a href="https://www.edweek.org/policy-politics/no-child-left-behind-an-overview/2015/04">https://www.edweek.org/policy-politics/no-child-left-behind-an-overview/2015/04</a></p>
<p style="padding-left: 2em; text-indent: -2em;">National Center for Education Statistics. (2015). <em>School district geographic relationship files.</em> [Data set]. <a href="https://nces.ed.gov/programs/edge/Geographic/RelationshipFiles">https://nces.ed.gov/programs/edge/Geographic/RelationshipFiles</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Noelke, C., McArdle, N., Baek, M., Huntington, N., Huber, R., Hardy, E., & Acevedo-Garcia, D. (2020). <em>Child Opportunity Index 2.0 Technical Documentation.</em> <a href="http://diversitydatakids.org/research-library/research-brief/how-we-built-it">http://diversitydatakids.org/research-library/research-brief/how-we-built-it</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research</em>, <em>12</em>(85), 2825–2830. <a href="http://jmlr.org/papers/v12/pedregosa11a.html">http://jmlr.org/papers/v12/pedregosa11a.html</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Reardon, S. F., Ho, A. D., Shear, B. R., Fahle, E. M., Kalogrides, D., Jang, H., & Chavez, B. (2021). <em>Stanford Education Data Archive</em> (Version 4.1). [Data set]. Stanford University. <a href="http://purl.stanford.edu/db586ns4974">http://purl.stanford.edu/db586ns4974</a></p>
<p style="padding-left: 2em; text-indent: -2em;">Semuels, A. (2016, August 25). Good school, rich school; bad school, poor school: The inequality at the heart of America's education system. <em>The Atlantic.</em> <a href="https://www.theatlantic.com/business/archive/2016/08/property-taxes-and-unequal-schools/497333/">https://www.theatlantic.com/business/archive/2016/08/property-taxes-and-unequal-schools/497333/</a></p>
<p style="padding-left: 2em; text-indent: -2em;">United States Census Bureau. (2010). <em>2010: DEC redistricting data (PL 94-171).</em> [Data set]. <a href="https://data.census.gov/cedsci/table?q=Decennial%20Census%20population&g=0100000US%241400000&d=DEC%20Redistricting%20Data%20%28PL%2094-171%29&tid=DECENNIALPL2020.P1">https://data.census.gov/cedsci/table?q=Decennial%20Census%20population&g=0100000US%241400000&d=DEC%20Redistricting%20Data%20%28PL%2094-171%29&tid=DECENNIALPL2020.P1</a></p>
''', unsafe_allow_html=True)


Report.header('Appendix', anchor='appendix')

fig_cross_val = go.Figure(data=[go.Table(columnwidth = [100, 100, 200, 100],
                                             header=dict(values=list(cross_val_results_df.columns), 
                                                         fill_color='black', 
                                                         font=dict(color='white', size=16)), 
                                             cells=dict(values=[cross_val_results_df['Cluster'], cross_val_results_df['Cross-Val Iteration'], 
                                                                cross_val_results_df['Best Parameters'], cross_val_results_df['Best Score']], 
                                                        align=['left', 'left', 'left', 'right'], 
                                                        fill_color='grey', 
                                                        line_color='white',
                                                        font=dict(color='white', size=14),
                                                        format=[None, None, None, '.4f']))])
fig_cross_val.update_layout(
    autosize=False,
    margin_b=0,
    height=700,
    width=900,
    showlegend=False,
    title_text='Cross-Validation Grid Search Results',
)
Report.plotly_chart(fig_cross_val, sharing='streamlit')

seda_df['residuals'] = seda_df['cs_mn_all'] - seda_df['predictions']

fig_resid = px.scatter(seda_df, 
                       x='predictions', 
                       y='residuals', 
                       opacity=0.25, 
                       labels=dict(predictions='Predicted Values', residuals='Residuals'),
                       title='Residuals for All-Cluster Model',
                       height=800,
                       width=800
                       )
fig_resid.update_xaxes(showgrid=False)
fig_resid.update_yaxes(showgrid=False)
Report.plotly_chart(fig_resid, sharing='streamlit')