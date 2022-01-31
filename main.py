from sklearn.utils import class_weight
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
train_df = train_df.drop("change_type", axis=1).values


X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2)

weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y
)

# params = {
#     "objective": "multi:softmax",
#     "num_class": 6,
#     "max_depth": 10,
#     "alpha": 10 ** (-3),
#     "learning_rate": 0.1,
#     "n_estimators": 100,
# }


fix_params = {
    "max_depth": 8,
    "silent": False,
    "learning_rate": 0.1,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "objective": "multi:softprob",
    "n_estimators": 1100,
    "reg_alpha": 10**(-4),
    "reg_lambda": 15,
    "early_stopping_rounds": 12,
    "scoring": "f1_micro",
}


def xgb_f1(y, t):
    y_true = t.get_label()
    y = np.array([np.argmax(y[i]) for i in range(len(y))])
    res = 1 - f1_score(y_true, y, average="micro")
    print(res)
    return "f1", res


n_splits = 5
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=32)
xgb_clf = XGBClassifier(**fix_params)
val = np.zeros(train_df.shape[0])
pred_mat = np.zeros((test_x.shape[0], n_splits))
for fold_index, (train_index, val_index) in enumerate(folds.split(train_df, y)):
    print('Batch {} started...'.format(fold_index))
    bst = xgb_clf.fit(train_df[train_index], y[train_index],
                      eval_set=[(train_df[val_index], y[val_index])],
                      verbose=0,
                      eval_metric=xgb_f1
                      )
    val[val_index] = xgb_clf.predict(train_df[val_index])
    print('f1_score of this val set is {}'.format(
        f1_score(y[val_index], val[val_index], average='micro')))
    pred_mat[:, fold_index] = xgb_clf.predict(test_x)


def most_common(lst):
    return max(set(lst), key=lst.count)


pred = np.zeros(test_x.shape[0])
for i in range(test_x.shape[0]):
    pred[i] = most_common(list(pred_mat[i, :]))


# eval_set = [(X_train, y_train), (X_test, y_test)]
# xgb_clf = XGBClassifier(**fix_params)
# xgb_clf.fit(X_train, y_train, eval_set=eval_set,
#             eval_metric=xgb_f1, verbose=True)


# print("XGBoost model accuracy score: {0:0.4f}".format(accuracy_score(y_test, y_pred)))


# data_dmatrix = xgboost.DMatrix(data=train_df, label=y)
# params = {'colsample_bytree': 0.3, 'learning_rate': 0.1,
#           'max_depth': 5, 'alpha': 10}


dump(xgb_clf, "xgb_model.joblib")

pred_t = xgb_clf.predict(X_test)
print(f1_score(pred_t, y_test, average="micro"))

# pred_y = xgb_clf.predict(test_x)
pred_df = pd.DataFrame(pred, columns=["change_type"])
pred_df['change_type'] = pred_df['change_type'].apply(lambda x: int(x))

pred_df.to_csv("knn_sample_submission.csv", index=True, index_label="Id")
