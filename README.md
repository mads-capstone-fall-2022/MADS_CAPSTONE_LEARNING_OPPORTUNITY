# MADS Capstone Fall 2022 - Team Learning Opportunity
This project was completed for the Capstone course of the Master of Applied Data Science program at the University of Michigan's School of Information by Katie Andrews and Jayachandra Korrapati. 

The code in this repository will allow you to reproduce our analysis of 2 datasets:
- The Child Opportunity Index v2.0 (COI)
- The Standford Educational Data Archive v4.1 (SEDA)

We analyzed these datasets to explore US school district comparison by clustering school districts based on the COI factors, then by using the COI data to predict SEDA achievement scores for grade school reading and math.  

## Streamlit App
The results of our analysis are available in our Streamlit app: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://child-opportunity-mads.streamlit.app)

## Reproducing Our Results
In order to run these notebooks, particularly the ones using SEDA data, we used the University of Michigan's Advanced Research Computing resources.  Please be aware that you might not be able to run all the notebooks on a local system. 

### Cloning the Repository

```
git clone https://github.com/mads-capstone-fall-2022/MADS_CAPSTONE_LEARNING_OPPORTUNITY
```

### Installing Requirements

```
pip install -r requirements.txt
```

### Downloading Data
Some of the data we used, including the COI indicators, is already in the data_raw directory of this repository.  However, there were 2 files that were too large for GitHub.  Those will need to be dowloaded and placed in the data_raw directory before running the notebooks.

1. Stanford Educational Data Archive - 
	1. [SEDA Main Download Page](https://edopportunity.org/get-the-data/seda-archive-downloads/) - may require free registration
	2. File - seda_geodist_long_CS_4.1.csv (525 MB)
	3. [Direct File Download Link](https://stacks.stanford.edu/file/druid:db586ns4974/seda_geodist_long_cs_4.1.csv)
2. School district boundaries - 
	1. [National Center for Educational Statistics Geographic Files Page](https://nces.ed.gov/programs/edge/Geographic/DistrictBoundaries)
	2. File - Year 2021 Single Composite File (188 MB)
	3. [Direct File Download Link](https://nces.ed.gov/programs/edge/data/EDGESCHOOLDISTRICT_TL21_SY2021.zip)

Once downloaded, unzip the school district boundary file and place the `.shp` file and the SEDA `.csv` file in the data_raw directory of the cloned repository.

### Running the Notebooks
As several of the notebooks require outputs from earlier notebooks, they should be run in this order:
1. coi_data_cleaning - Establishes the train/test split of COI data, consolidates the COI data from census tracts into school districts, outputs train/test split keys for use with SEDA data and the split COI data
2. COI Kmeans clustering - 
3. COI DBSCAN clustering - 
4. seda_data_cleaning - Uses train/test split keys from coi_data_cleaning to split SEDA data and filter it to use only 2016-2018 data
5. model_training - Uses the train/test split COI and SEDA data and the learnings from the clustering notebooks to create a K-Means cluster model of the COI data, trains a variety of predictive models on the COI data to predict the overall mean scores from the SEDA data
6. display_items - Uses the data outputs of model_training to reproduce the figures and tables used in our Streamlit app

