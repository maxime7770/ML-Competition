import pandas as pd
import xgboost
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,
                            max_depth=12,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            )
n_data_max = 20000

print("Loading data...")
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')
Y = train_df["change_type"][:n_data_max]
X = train_df.drop("change_type", 1)
X = X[['area',
       'length', 'boxcox_area', 'boxcox_length', 'area/length**2',
    #    'elongation', 'minx', 'miny', 'maxx', 'maxy', 'centroid_x',
    #    'centroid_y', 'height', 'width', 'nb_points', 'diff_area', 'is_convex',
    #    'centroid_dist', 'length/width', 'Dense Urban', 'Industrial', 'None',
    #    'Rural', 'Sparse Urban', 'Urban Slum', 'Barren Land', 'Coastal',
    #    'Dense Forest', 'Desert', 'Farms', 'Grass Land', 'Hills', 'Lakes',
       'None.1', 'River', 'Snow', 'Sparse Forest']][:n_data_max]

print("Cross val ...")
n_folds = 10
scores = cross_val_score(rf, X, Y , cv = n_folds)
m = scores.mean()
std = scores.std()
print(f"CV score: {round(m,2)}% +/- {round(std, 2)}%")
