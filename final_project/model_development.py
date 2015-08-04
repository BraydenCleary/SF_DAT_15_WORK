# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:22:53 2015

@author: braydencleary
"""

# link to train data: https://www.dropbox.com/s/xgas3ao2as5n5eo/crime_train.csv?dl=0
# link to test data: https://www.dropbox.com/s/jt5wvpnlhh9s4yf/crime_test.csv?dl=0
# scp -i  /Users/braydencleary/.ssh/braydencleary.pem /Users/braydencleary/Desktop/crime_train.csv ec2-user@ec2-54-164-20-61.compute-1.amazonaws.com:~/
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime

# Read in dataset
crime_train = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_with_features.csv')
# drop columns where zip is 0 (gps coordinates were whack)
crime_train = crime_train[crime_train.zip != 0]

X = c[['zip', 'CensusAgeMedianAge', 'CensusAgePercent10To14', 'DayOfWeek_Friday', 'DayOfWeek_Monday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN', 'is_weekend', 'time_of_day_bucket', 'day_of_month', 'month_of_year', 'year']]
y = c['Category']


##### need to redo everything below given my above dataset #####

test_knn = KNeighborsClassifier(n_neighbors=3)
test_knn.fit(test_X, test_Y )
test['predictions'] = test_knn.predict(test_X)

binary_columns = np.array(with_dummies_train.columns[6:])

X = with_dummies_train[binary_columns]
y = with_dummies_train['Category']
#features_test  = with_dummies_test[binary_columns]
#response_test  = with_dummies_test['Category']

#Train test split
features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=1, train_size=.33)

#Weekday Columns
weekday_columns = np.array(with_dummies_train.columns[6:13])
X = with_dummies_train[weekday_columns]

features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train, response_train)
knn.score(features_test, response_test) # .09 accuracy

####
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(features_train, response_train)
knn.score(features_test, response_test) #.19 accuracy

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


