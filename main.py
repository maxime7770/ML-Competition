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


train_df=pd.read_csv('train_df.csv')
test_df=pd.read_csv('test_df.csv')

test_x = test_df.values
y = train_df['change_type'].values
train_df = train_df.drop('change_type', axis=1)


X_train, X_test, y_train, y_test = train_test_split(train_df, y)

params = {
    'objective': 'multi:softmax',
    'num_class': 6,
    'max_depth': 10,
    'alpha': 10**(-3),
    'learning_rate': 0.1,
    'n_estimators': 100,
    'verbosity': 1
}

xgb_clf = XGBClassifier(**params)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

#xgb_clf.fit(X_train, y_train)
print(1)
scores = cross_val_score(xgb_clf, train_df, y,
                         scoring='f1', cv=cv, n_jobs=-1, verbose=10)
print(2)
print(scores)
xgb_clf.fit(train_df, y)
y_pred = xgb_clf.predict(X_test)
print('XGBoost model accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred)))


# data_dmatrix = xgboost.DMatrix(data=train_df, label=y)
# params = {'colsample_bytree': 0.3, 'learning_rate': 0.1,
#           'max_depth': 5, 'alpha': 10}

# xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
#             num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
# X_train, X_test, y_train, y_test = train_test_split(train_df, y)
# print(xgb_cv.head())

dump(xgb_clf, 'xgb_model.joblib')
pred_y = xgb_clf.predict(test_x)


pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')
