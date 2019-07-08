import numpy as np
import pandas as pd
import pickle

import lightgbm as lgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

train = pd.DataFrame()
seasons = [2013,2014,2015,2016,2017,2018,2019]
for season in seasons:
    print("Adding new season...")
    sea_df = pd.read_csv('./data/seasons/'+str(season)+'.csv')
    train = pd.concat([train, sea_df],axis=0)
print(len(train))
y= train.Result.values.astype(int)
print(np.mean(y))
X = train.drop(columns=['Unnamed: 0','Start_Date','Result'])
X.Round = X.Round.replace('R1',1)
X.Round = X.Round.replace('R2',2)
X.Round = X.Round.replace('R3',3)
X.Round = X.Round.replace('R4',4)

# X = X[['Elo','P1_DS','P2_DS']]

print("Shuffling")
X = X.values
X = shuffle(X)

print("Splitting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num_train, num_feature = X_train.shape
lgb_train = lgb.Dataset(X_train, y_train,
                        free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                       free_raw_data=False)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.025,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

feature_name = ['feature_' + str(col) for col in range(num_feature)]
#
# estimator = lgb.LGBMRegressor()
#
# gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm = lgb.LGBMRegressor(num_leaves=25,
                        learning_rate=0.01,
                        n_estimators=15)
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10,
#                 valid_sets=lgb_train,  # eval training data
#                 feature_name=feature_name,
#                 categorical_feature=[21])

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)
# print('Best parameters found by grid search are:', gbm.best_params_)

y_pred = gbm.predict(X_test)

print(y_pred)
print(np.mean(y_pred))
print(np.mean(y_test))
# train['Elo_Bi'] = np.where(train['Elo']>=0.5, 1, 0)
# train['Glicko_Bi'] = np.where(train['Glicko']>=0.5, 1, 0)
# print(log_loss(y, train.Elo.values))
# print(log_loss(y, train.Glicko.values))
print(log_loss(y_test, y_pred))
pickle.dump(gbm, open('model.sav', 'wb'))
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print('Feature importances:', list(gbm.feature_importances_))




#
