# Kaggle competition as part of the Machine Learning elective course at CentraleSupélec


**Ranked 1st among 76 teams**

**[Link of the competition](https://www.kaggle.com/competitions/centralesypelec-ml2022-course)**


The folder contains all the necessary code to create all the features and get the scores we had for submissions.


# Authors

Wolf Maxime, Palaric Aymeric, Levy Guillaume, Boulet Timothé


# Description of the folder

capital_countries.ipynb: creates the features like the country where the polygon is or distance to the nearest capial

convexity.py: contains a function that tests the convexity of a polygon (usef in preprocessing.py)

eval_model.ipynb: used for training of the model and predictions 

extract_features_dates.ipynb: creation of the dates features (duration in days between today and the date, duration between
2 consecutive dates and duration to make an advancement between two status, etc.)

fourier_transform.ipynb: build fourier coefficients and fourier power as explained in the report 

nearest_buildings.ipynb: contains 3 important functions that add the features of the kNN of the polygons, the mean of the features of the kNN polygons, and the area of the minimal polygon that contains the centroids of the kNN 

preprocessing.py: performs basic preprocessing (see report)

utils.py: regroup all the contents of the other files to add all the features in the same dataframe, it is used to choose what features to add for the training of our model

other_models.ipynb: was used to create, train and compare different models 


# How does it work?

To generate the features: run 
- preprocessing.py 
- extract_features_dates.ipynb
- capital_countries.ipynb
- fourier_transform.ipynb
- nearest_buildings.ipynb

Then run:
- eval_model.ipynb (this will call load_data in utils.py to get all the features in the same dataframe)

