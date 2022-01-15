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
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost
from joblib import dump, load
from datetime import datetime


train_df = pd.read_csv("train_df.csv", index_col=0)
test_df = pd.read_csv("test_df.csv", index_col=0)
# train_df = train_df.drop(
#     ["index", "diff1", "diff2", "diff3", "diff4", "area", "length"], axis=1
# )
# test_df = test_df.drop(
#     ["index", "diff1", "diff2", "diff3", "diff4", "area", "length"], axis=1
# )

train_df = train_df.drop("index", axis=1)
test_df = test_df.drop("index", axis=1)

test_x = test_df.values
y = train_df["change_type"].values
train_df = train_df.drop("change_type", axis=1)


X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2)

# params = {
#     "objective": "multi:softmax",
#     "num_class": 6,
#     "max_depth": 10,
#     "alpha": 10 ** (-3),
#     "learning_rate": 0.1,
#     "n_estimators": 100,
# }


# cv_params = {
#     "max_depth": [1, 2, 3, 4, 5, 6],
#     "min_child_weight": [1, 2, 3, 4],
# }  # parameters to be tries in the grid search
fix_params = {
    "max_depth": 6,
    "silent": False,
    #    "scale_pos_weight": 1,
    "learning_rate": 0.1,
    #    "min_child_weight": 4,
    "colsample_bytree": 0.7,
    "subsample": 0.6,
    "objective": "multi:softprob",
    "n_estimators": 1500,
    "reg_alpha": 10**(-4),
    "reg_lambda": 10,
    "early_stopping_rounds": 10,
    "scoring": "f1_micro",
    "gpu_id": 0,
    "tree_method": "gpu_hist"
}  # other parameters, fixed for the moment


def xgb_f1(y, t):
    y_true = t.get_label()
    print(y_true)
    y = np.array([np.argmax(y[i]) for i in range(len(y))])
    print(y)
    return "f1", 1 - f1_score(y_true, y, average="micro")


# xgb_clf = GridSearchCV(XGBClassifier(**fix_params), cv_params, scoring="f1_micro", cv=5)
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_clf = XGBClassifier(**fix_params)
xgb_clf.fit(X_train, y_train, eval_set=eval_set,
            eval_metric=xgb_f1, verbose=True)


# print("XGBoost model accuracy score: {0:0.4f}".format(accuracy_score(y_test, y_pred)))


# data_dmatrix = xgboost.DMatrix(data=train_df, label=y)
# params = {'colsample_bytree': 0.3, 'learning_rate': 0.1,
#           'max_depth': 5, 'alpha': 10}


dump(xgb_clf, "xgb_model.joblib")

pred = xgb_clf.predict(X_test)
print(f1_score(pred, y_test, average="micro"))

pred_y = xgb_clf.predict(test_x)
pred_df = pd.DataFrame(pred_y, columns=["change_type"])
pred_df.to_csv("knn_sample_submission.csv", index=True, index_label="Id")
