from xgboost import XGBClassifier, cv
import geopandas as gpd
import pandas as pd
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
import xgboost
from joblib import dump, load
from datetime import datetime


train_df = gpd.read_file("train.geojson", index_col=0)
test_df = gpd.read_file("test.geojson", index_col=0)

train_df = train_df.drop("index", axis=1)
test_df = test_df.drop("index", axis=1)

x = "date1"
train_df["hour" + "_" + x] = train_df[x].apply(
    lambda x: datetime.strptime(x, "%d-%m-%Y").month
)

print(train_df["hour_date1"])

# t = train_df[
#     [
#         "change_status_date1",
#         "change_status_date2",
#         "change_status_date3",
#         "change_status_date4",
#         "change_status_date5",
#     ]
# ].values

# print([i for i in range(t.shape[0]) if "Construction Done" not in t[i, :]])

# finished = []
# for i in range(t.shape[0]):
#     if "Construction Done" not in t[i, :]:
#         finished.append(1)
#     else:
#         finished.append(0)
# train_df["finished"] = np.array(finished)
# print(train_df["finished"])
# dates = train_df[["date1", "date2", "date3", "date4", "date5"]]
# date_done = []
# for i in range(t.shape[0]):
#     j = 0
#     while t[i, j] != "Construction Done":
#         j += 1
#     k = "date" + str(j)
#     date_done.append(dates[k].values[i])

# print(date_done)
