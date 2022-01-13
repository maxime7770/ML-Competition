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
from scipy.stats import boxcox


change_type_map = {
    "Demolition": 0,
    "Road": 1,
    "Residential": 2,
    "Commercial": 3,
    "Industrial": 4,
    "Mega Projects": 5,
}
# to delete last classes

change_type_map = {
    "Demolition": 0,
    "Road": 1,
    "Residential": 2,
    "Commercial": 3,
    "Industrial": 4
}

train_df = gpd.read_file("train.geojson", index_col=0)
test_df = gpd.read_file("test.geojson", index_col=0)


# trying to delete last classes


train_df = train_df.drop(
    train_df[(train_df["change_type"] == "Mega Projects")].index)


# and thus drop if status_sate=excavation because it is essentially in the 5th class

# train_df = train_df.drop(
#     train_df[
#         (train_df["change_status_date1"] == "Excavation")
#         | (train_df["change_status_date2"] == "Excavation")
#         | (train_df["change_status_date3"] == "Excavation")
#         | (train_df["change_status_date4"] == "Excavation")
#         | (train_df["change_status_date5"] == "Excavation")
#     ].index
# )

# test_df = test_df.drop(
#     test_df[
#         (test_df["change_status_date1"] == "Excavation")
#         | (test_df["change_status_date2"] == "Excavation")
#         | (test_df["change_status_date3"] == "Excavation")
#         | (test_df["change_status_date4"] == "Excavation")
#         | (test_df["change_status_date5"] == "Excavation")
#     ].index
# )

# dropping rows if "Na" values in change_status_datei

# train_df = train_df.drop(
#     train_df[
#         (train_df["change_status_date1"] == "Na")
#         | (train_df["change_status_date2"] == "Na")
#         | (train_df["change_status_date3"] == "Na")
#         | (train_df["change_status_date4"] == "Na")
#         | (train_df["change_status_date5"] == "Na")
#     ].index
# )
# test_df = test_df.drop(
#     test_df[
#         (test_df["change_status_date1"] == "Na")
#         | (test_df["change_status_date2"] == "Na")
#         | (test_df["change_status_date3"] == "Na")
#         | (test_df["change_status_date4"] == "Na")
#         | (test_df["change_status_date5"] == "Na")
#     ].index
# )


# creating vectors of 0 or 1 if the construction finished before the 5 days or not:
# t_train = train_df[
#     [
#         "change_status_date1",
#         "change_status_date2",
#         "change_status_date3",
#         "change_status_date4",
#         "change_status_date5",
#     ]
# ].values

# finished_train = []
# for i in range(t_train.shape[0]):
#     if "Construction Done" not in t_train[i, :]:
#         finished_train.append(1)
#     else:
#         finished_train.append(0)
# train_df["finished"] = np.array(finished_train)

# t_test = test_df[
#     [
#         "change_status_date1",
#         "change_status_date2",
#         "change_status_date3",
#         "change_status_date4",
#         "change_status_date5",
#     ]
# ].values

# finished_test = []
# for i in range(t_test.shape[0]):
#     if "Construction Done" not in t_test[i, :]:
#         finished_test.append(1)
#     else:
#         finished_test.append(0)
# test_df["finished"] = np.array(finished_test)


# adding time differences

train_df["diff1"] = (
    train_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date1"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
train_df["diff2"] = (
    train_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
train_df["diff3"] = (
    train_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
train_df["diff4"] = (
    train_df["date5"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - train_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)


test_df["diff1"] = (
    test_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date1"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
test_df["diff2"] = (
    test_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date2"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
test_df["diff3"] = (
    test_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date3"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)
test_df["diff4"] = (
    test_df["date5"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
    - test_df["date4"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
).apply(lambda x: x.days)


for x in ["date1", "date2", "date3", "date4", "date5"]:
    # weekday
    train_df["weekday" + "_" + x] = train_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").weekday()
    )
    test_df["weekday" + "_" + x] = test_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").weekday()
    )

    # month
    train_df["month" + "_" + x] = train_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").month
    )
    test_df["month" + "_" + x] = test_df[x].apply(
        lambda x: datetime.strptime(x, "%d-%m-%Y").month
    )


# adding perimters and areas of polygons

train_df["area"] = train_df["geometry"].area
train_df["length"] = train_df["geometry"].length


# a = train_df["area"]
# b = train_df["length"]

# train_df["area"] = (a - a.min()) / (a.max() - a.min())
# train_df["length"] = (b - b.min()) / (b.max() - b.min())

test_df["area"] = test_df["geometry"].area
test_df["length"] = test_df["geometry"].length


# a = test_df["area"]
# b = test_df["length"]

# test_df["area"] = (a - a.min()) / (a.max() - a.min())
# test_df["length"] = (b - b.min()) / (b.max() - b.min())

train_df = train_df.drop(train_df[train_df["area"] == 0].index)
test_df = test_df.drop(test_df[test_df["area"] == 0].index)


# 1 over area or length:

train_df["1/area"] = 1/train_df["area"]
test_df["1/area"] = 1/test_df["area"]

train_df["1/length"] = 1/train_df["length"]
test_df["1/length"] = 1/test_df["length"]

# boxcox transformation

train_df["boxcox_area"], par = boxcox(train_df["area"])
test_df["boxcox_area"], par = boxcox(test_df["area"])

train_df["boxcox_length"], par = boxcox(train_df["length"])
test_df["boxcox_length"], par = boxcox(test_df["length"])

# square root transformation

train_df["sqrt_area"] = np.sqrt(train_df["area"])
test_df["sqrt_area"] = np.sqrt(test_df["area"])


# length squared

train_df["squared_length"] = train_df["length"]**2
test_df["squared_length"] = test_df["length"]**2


# length over area :

train_df["length/area"] = train_df["length"]/train_df["area"]
test_df["length/area"] = test_df["length"]/test_df["area"]

# length * area

train_df["length*area"] = train_df["length"]*train_df["area"]
test_df["length*area"] = test_df["length"]*test_df["area"]

train_df = train_df.drop("geometry", axis=1)
test_df = test_df.drop("geometry", axis=1)

dates = ["date1", "date2", "date3", "date4", "date5"]
for d in dates:
    train_df = train_df.drop(d, axis=1)
    test_df = test_df.drop(d, axis=1)

col_str = [
    "change_status_date1",
    "change_status_date2",
    "change_status_date3",
    "change_status_date4",
    "change_status_date5",
]
le = LabelEncoder()
train_df[col_str] = train_df[col_str].apply(le.fit_transform)
test_df[col_str] = test_df[col_str].apply(le.fit_transform)

le2 = LabelEncoder()
train_df[["urban_types"]] = train_df[["urban_types"]].apply(le2.fit_transform)
test_df[["urban_types"]] = test_df[["urban_types"]].apply(le2.fit_transform)


le3 = LabelEncoder()
train_df[["area"]] = train_df[["area"]].apply(le3.fit_transform)
test_df[["area"]] = test_df[["area"]].apply(le3.fit_transform)

le4 = LabelEncoder()
train_df[["length"]] = train_df[["length"]].apply(le4.fit_transform)
test_df[["length"]] = test_df[["length"]].apply(le4.fit_transform)


train_df["geography_types"] = train_df["geography_types"].apply(
    lambda x: x.split(","))
test_df["geography_types"] = test_df["geography_types"].apply(
    lambda x: x.split(","))


train_df = pd.concat(
    [train_df, train_df["geography_types"].str.join("|").str.get_dummies()], axis=1
)
test_df = pd.concat(
    [test_df, test_df["geography_types"].str.join("|").str.get_dummies()], axis=1
)

train_df = train_df.drop("geography_types", axis=1)
test_df = test_df.drop("geography_types", axis=1)


train_df["change_type"] = train_df["change_type"].apply(
    lambda x: change_type_map[x])


train_df.to_csv("train_df.csv")
test_df.to_csv("test_df.csv")
