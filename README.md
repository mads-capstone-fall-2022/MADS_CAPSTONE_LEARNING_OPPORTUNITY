# Evaluating US School District Achievement Scores Based on Community Resource Levels
## MADS Capstone Fall 2022 - Team Learning Opportunity

This project was completed for the Capstone course of the Master of Applied Data Science program at the University of Michigan's School of Information by Kathryn Andrews and Jayachandra Korrapati. 

The code in this repository will allow you to reproduce our analysis of 2 datasets:
- The [Child Opportunity Index v2.0](https://data.diversitydatakids.org/dataset/coi20-child-opportunity-index-2-0-database) (COI) from [datadiversitykids.org](https://www.diversitydatakids.org/)
- The [Stanford Educational Data Archive v4.1](https://edopportunity.org/get-the-data/seda-archive-downloads/) (SEDA) from the [Educational Opportunity Project at Stanford University](https://edopportunity.org/)

We analyzed these datasets to explore US school district comparison by clustering school districts based on the COI factors, then by using the COI data to predict SEDA achievement scores for grade school reading and math.  

## Streamlit App
The results of our analysis are available in our Streamlit app: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://child-opportunity-mads.streamlit.app)

## Reproducing Our Results
In order to run these notebooks, particularly the ones using SEDA data, we used the University of Michigan's Advanced Research Computing resources.  Please be aware that you might not be able to run all the notebooks on a local system.  Also, the model_training notebook has a parametere in the first cell: n_jobs.  This is used by several models in the notebook and should be set to the number of cores you have available for multi-threaded jobs.

### Cloning the Repository

```
git clone https://github.com/mads-capstone-fall-2022/MADS_CAPSTONE_LEARNING_OPPORTUNITY
```

### Installing Requirements

```
pip install -r requirements.txt --user
```

### Downloading Data
Some of the data we used, including the COI indicators, is already in the data_raw directory of this repository.  However, there were 2 files that were too large for GitHub.  Those will need to be dowloaded and placed in the data_raw directory before running the notebooks.

1. Stanford Educational Data Archive - 
	1. [SEDA Main Download Page](https://edopportunity.org/get-the-data/seda-archive-downloads/) - may require free registration
	2. File - seda_geodist_long_CS_4.1.csv (525 MB)
	3. [Direct File Download Link](https://stacks.stanford.edu/file/druid:db586ns4974/seda_geodist_long_cs_4.1.csv)
2. School district boundaries - 
	1. [National Center for Educational Statistics Geographic Files Page](https://nces.ed.gov/programs/edge/Geographic/DistrictBoundaries)
	2. File - Year 2015 Single Composite File (138 MB)
	3. [Direct File Download Link](https://nces.ed.gov/programs/edge/data/SCHOOLDISTRICT_SY1314_TL15.zip)

Once downloaded, place the entire school district boundary `.zip` file and the SEDA `.csv` file in the data_raw directory of the cloned repository.

### Running the Notebooks
As several of the notebooks require outputs from earlier notebooks, they should be run in this order:


| Order | Notebook Name | Content Notes |
| --- | --- | --- |
| 1 | `coi_data_cleaning` | Establishes the train/test split of COI data, consolidates the COI data from census tracts into school districts, outputs train/test split keys for use with SEDA data and the split COI data. |
| 2 | `COI Kmeans clustering` | This notebook has code which uses Kmeans clustering to cluster school districts from the COI information. |
| 3 | `COI DBSCAN clustering` | This notebook has code which uses DBSCAN clustering to cluster school districts from the COI information. We used the results of Kmeans clustering eventually, so this notebook is more for reference. |
| 4 | `seda_data_cleaning` | Uses train/test split keys from coi_data_cleaning to split SEDA data and filter it to use only 2016-2018 data. |
| 5 | `model_training` | Uses the train/test split COI and SEDA data and the learnings from the clustering notebooks to create a K-Means cluster model of the COI data,trains a variety of predictive models on the COI data to predict the overall mean scores from the SEDA data. |
| 6 | `display_items` | Uses the data outputs of model_training to reproduce the figures and tables used in our Streamlit app.|


Three caveats about the notebooks:
1. In `model_training`, the cross-validation code will generate slightly different results every time it is run, due to the influence of randomness in the splits.  Unlike other models, I did not use the random_state = 42 setting, because that made all the splits the same as each other, defeating the purpose of cross-validation.
2. Two files (`model_results.csv` and `cross_val_results.csv`) used in `display_items` were manually typed to get all the model and cross-validation results in a clean, tabular format for display in Streamlit.  These prepared files have been placed in the data_display folder.
3. In `display_items`, the line `pio.renderers.default='iframe'` in the first cell may need to be commented out for plots to be displayed.  This line is required in some environments but not in others.