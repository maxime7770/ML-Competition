import sys
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


n_data_max = 99999999
n_folds = 5
scoring = "accuracy"
add_knn_mean = False
add_knn_concat = False

print("Loading data...")
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')
idx = idx = np.random.permutation(train_df.index)  #Shuffle train dataset

Y = train_df["change_type"].reindex(idx)[:n_data_max]
X = train_df.drop("change_type", 1).reindex(idx)[:n_data_max]

X = X[[
      'area', 'length', 'area/length**2',
      'elongation', 'centroid_x',
      'centroid_y', 'height', 'width', 'nb_points', 'diff_area', 'is_convex',
      'centroid_dist', 'length/width', 'Dense Urban', 'Industrial', 'None',
      'Rural', 'Sparse Urban', 'Urban Slum', 'Barren Land', 'Coastal',
      'Dense Forest', 'Desert', 'Farms', 'Grass Land', 'Hills', 'Lakes',
      'None.1', 'River', 'Snow', 'Sparse Forest']]

if add_knn_mean:
      X_knn_aug = pd.read_csv('train_df_knn_aug.csv')
      X_knn_aug = X_knn_aug[[
         'knn_area', 'knn_length', 'knn_area/length**2', 'knn_elongation',
         'knn_centroid_x', 'knn_height', 'knn_width', 'knn_nb_points',
         'knn_centroid_dist', 'knn_length/width', 'knn_Dense Urban',
         'knn_Industrial', 'knn_None', 'knn_Rural', 'knn_Sparse Urban',
         'knn_Urban Slum', 'knn_Barren Land', 'knn_Coastal', 'knn_Dense Forest',
         'knn_Desert', 'knn_Farms', 'knn_Grass Land', 'knn_Hills', 'knn_Lakes',
         'knn_None.1', 'knn_River', 'knn_Snow', 'knn_Sparse Forest'
      ]].reindex(idx)[:n_data_max]
      X = pd.concat([X, X_knn_aug], axis=1)
            
if add_knn_concat:
      X_knn_aug = pd.read_csv('train_df_knn_concat.csv').reindex(idx)[:n_data_max]
      X = pd.concat([X, X_knn_aug], axis=1)

print("Cross val ...")
scores = cross_val_score(rf, X, Y , cv = n_folds, 
                        scoring = scoring,
                        )
m = scores.mean()
std = scores.std()
print(f"CV score for {scoring}: {round(100*m,2)}% +/- {round(100*std, 2)}%")
