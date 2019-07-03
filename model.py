import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
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

X = pd.DataFrame()
# seasons = [2014,2015,2016,2017,2018,2019]
seasons = [2004,2005,2006,2007,2008,2009]
for season in seasons:
    print("Adding new season...")
    sea_df = pd.read_csv('./data/seasons/'+str(season)+'.csv')
    X = pd.concat([X, sea_df],axis=0)
print(len(X))
y= X.Result.values.astype(int)
print("AVERAGE: ", np.mean(y))
raise ValueError
X = X.drop(columns=['Unnamed: 0','Start_Date','Result'])
X.Round = X.Round.replace('R1',1)
X.Round = X.Round.replace('R2',2)
X.Round = X.Round.replace('R3',3)
X.Round = X.Round.replace('R4',4)

print("Shuffling")
X = X.values
X = shuffle(X)
X = shuffle(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict_proba(X_test)

# print(log_loss(y, X.Elo.values))
# print(log_loss(y, X.Glicko.values))
# print(log_loss(y, X.Average.values))
print(log_loss(y_test, y_pred))
pickle.dump(logreg, open('model.sav', 'wb'))




#
