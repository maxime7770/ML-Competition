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


change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
                   'Mega Projects': 5}

train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

# adding time differences

train_df['diff1'] = (train_df['date2'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                     - train_df['date1'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())
train_df['diff2'] = (train_df['date3'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                     - train_df['date2'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())
train_df['diff3'] = (train_df['date4'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                     - train_df['date3'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())
train_df['diff4'] = (train_df['date5'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                     - train_df['date4'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())


test_df['diff1'] = (test_df['date2'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                    - test_df['date1'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())
test_df['diff2'] = (test_df['date3'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                    - test_df['date2'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())
test_df['diff3'] = (test_df['date4'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                    - test_df['date3'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())
test_df['diff4'] = (test_df['date5'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))
                    - test_df['date4'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y"))).apply(lambda x: x.total_seconds())


# adding perimters and areas of polygons

train_df['area'] = train_df['geometry'].area
train_df['length'] = train_df['geometry'].length


a = train_df['area']
b = train_df['length']

train_df['area'] = (a-a.min())/(a.max()-a.min())
train_df['length'] = (b-b.min())/(b.max()-b.min())

test_df['area'] = test_df['geometry'].area
test_df['length'] = test_df['geometry'].length


a = test_df['area']
b = test_df['length']

test_df['area'] = (a-a.min())/(a.max()-a.min())
test_df['length'] = (b-b.min())/(b.max()-b.min())


train_df = train_df.drop('geometry', axis=1)
test_df = test_df.drop('geometry', axis=1)

dates = ['date1', 'date2', 'date3', 'date4', 'date5']
for d in dates:
    train_df = train_df.drop(d, axis=1)
    test_df = test_df.drop(d, axis=1)

col_str = ['change_status_date1', 'change_status_date2',
           'change_status_date3', 'change_status_date4', 'change_status_date5']
le = LabelEncoder()
train_df[col_str] = train_df[col_str].apply(le.fit_transform)
test_df[col_str] = test_df[col_str].apply(le.fit_transform)

le2 = LabelEncoder()
train_df[['urban_types']] = train_df[['urban_types']].apply(le2.fit_transform)
test_df[['urban_types']] = test_df[['urban_types']].apply(le2.fit_transform)


train_df['geography_types'] = train_df['geography_types'].apply(
    lambda x: x.split(','))
test_df['geography_types'] = test_df['geography_types'].apply(
    lambda x: x.split(','))


train_df = pd.concat([train_df, train_df['geography_types'].str.join(
    '|').str.get_dummies()], axis=1)
test_df = pd.concat([test_df, test_df['geography_types'].str.join(
    '|').str.get_dummies()], axis=1)

train_df = train_df.drop('geography_types', axis=1)
test_df = test_df.drop('geography_types', axis=1)


train_df['change_type'] = train_df['change_type'].apply(
    lambda x: change_type_map[x])


train_df.to_csv('train_df')
test_df.to_csv('test_df')
