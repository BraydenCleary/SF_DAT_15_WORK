# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:22:53 2015

@author: braydencleary
"""

# link to train data: https://www.dropbox.com/s/xgas3ao2as5n5eo/crime_train.csv?dl=0
# link to test data: https://www.dropbox.com/s/jt5wvpnlhh9s4yf/crime_test.csv?dl=0
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

# For each predictor, convert to a binary column
crime_train = pd.read_csv("../../../crime_train.csv")
#crime_test  = pd.read_csv("../../../crime_test.csv")

crime_train.rename(columns={'X': 'Long', 'Y': 'Lat'}, inplace=True)
#crime_test.rename(columns={'X': 'Long', 'Y': 'Lat'}, inplace=True)

with_dummies_train = pd.get_dummies(crime_train, columns=['DayOfWeek', 'PdDistrict', 'Resolution'])
#with_dummies_test = pd.get_dummies(crime_test, columns=['DayOfWeek', 'PdDistrict'])

binary_columns = np.array(with_dummies_train.columns[6:])

X = with_dummies_train[binary_columns]
y = with_dummies_train['Category']
#features_test  = with_dummies_test[binary_columns]
#response_test  = with_dummies_test['Category']

features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(features_train, response_train)
knn.score(features_test, response_test) #.19 accuracy

####

weekday_columns = np.array(with_dummies_train.columns[6:13])
X = with_dummies_train[weekday_columns]

features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(features_train, response_train)
knn.score(features_test, response_test) # .09 accuracy

####

# Use grid search
knn = KNeighborsClassifier()
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

####

ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
X = with_dummies_train[binary_columns]
y = with_dummies_train['Category']

features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=1)

ctree.fit(features_train, response_train)
preds = ctree.predict(features_test)
metrics.accuracy_score(response_test, preds) # .28

####

X = with_dummies_train[binary_columns]
y = with_dummies_train['Category']

ctree = tree.DecisionTreeClassifier(random_state=1)
depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5)
grid.fit(X, y)

#grid_mean_scores = [result[1] for result in grid.grid_scores_]
#plt.figure()
#plt.plot(depth_range, grid_mean_scores)
#plt.hold(True)
#plt.grid(True)
#plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         #markerfacecolor='None', markeredgecolor='r')

grid.best_score_ # 33.09

#### 
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

X = with_dummies_train[binary_columns]
y = with_dummies_train['Category']

features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=1)
nb.fit(features_train, response_train)
preds = nb.predict(features_test)

metrics.accuracy_score(response_test, preds) #.32

#### 

with_dummies_train['weapon_present'] = crime_train.apply(lambda x: 1 if any(word in ['knife', 'gun', 'weapon'] for word in x['Descript'].lower().split(' ')) else 0, axis=1)

binary_columns = np.array(with_dummies_train.columns[6:])
X = with_dummies_train[binary_columns]
y = with_dummies_train['Category']

features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=1)
nb.fit(features_train, response_train)
preds = nb.predict(features_test)

metrics.accuracy_score(response_test, preds) #.34


