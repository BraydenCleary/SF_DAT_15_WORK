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

# columns ['Unnamed: 0', 'Dates', 'Category', 'Descript', 'DayOfWeek','PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'zip','CensusAgeMedianAge', 'CensusAgePercent10To14', 'LatitudeDeg','CensusHouseholdsWithOver65','CensusHouseholdsPercentHouseholdsWithUnder18', 'CensusAgeOver18','CensusHouseholdsPercentFamilyKidsUnder18','CensusRacePercentIndian', 'CensusRacePercentAsianOnly','CensusAgeUnder18', 'CensusTotalPopulation','CensusAgePercent45To54', 'CensusAgePercentOver65Female','CensusGeoId', 'CensusAgePercentOver18Female','CensusRacePercentHawaiianPIOnly','CensusHouseholdsPercentHouseholderOver65', 'CensusAge18To19','CensusSexMale', 'CensusEsriId', 'CensusAgeOver21','CensusAgePercent15To17', 'CensusAgePercent15To19','CensusHispanicMexican', 'CensusHispanicPercentNonHispanic','CensusAge0To4', 'CensusSexFemale', 'LongitudeDeg','CensusAgePercent5To9', 'CensusAge20To24','CensusHouseholdsHouseholderOver65', 'CensusRaceWhiteOnly','CensusAge60To64', 'CensusHouseholdsPercentSameSexPartner','CensusAge25To44', 'CensusRacePercentOtherOnly', 'MinDistanceKm','CensusHouseholdsPercentLivingAlone','CensusHouseholdsPercentFamilies', 'FeatureClassCode','CensusHouseholdsUnmarriedPartner', 'CensusHouseholdsNonFamily','CensusRaceBlackOnly', 'WaterAreaSqM', 'CensusAgePercentOver65','CensusAgePercentOver62', 'FunctionalStatus','CensusHispanicTotalPopulation', 'CensusHispanicWhiteNonHispanic','CensusHispanicPercentOtherHispanic','CensusTotalPopulationPercentChange2000To2010','CensusRaceOneRaceOnly', 'CensusAgePercent65To74','CensusAge45To54', 'CensusAgePercent55To59', 'AskGeoId','CensusAge15To19', 'CensusAgeOver65Male', 'CensusAge15To17','CensusAgePercentOver18', 'CensusAgePercentOver65Male','CensusHouseholdsTotal', 'CensusAgePercent45To64','CensusRaceAsianOnly', 'CensusAgeOver65Female', 'GeoId','CensusAge45To64', 'CensusGeoCode', 'CensusStateAbbreviation','CensusAgePercentOver85', 'CensusAge10To14','CensusHouseholdsPercentFemaleHouseholder', 'CensusAge55To59','CensusHouseholdsSameSexPartner', 'CensusRaceMultiRace','CensusHispanicPercentMexican', 'CensusHispanicPuertoRican','CensusAgePercent60To64', 'CensusSexPercentFemale','CensusHouseholdsMarriedCoupleKidsUnder18','CensusHispanicPercentWhiteNonHispanic','CensusTotalPopulationIn2000','CensusHouseholdsPercentMarriedCouple','CensusHouseholdsHouseholdsWithUnder18','CensusHouseholdsPercentSingleMoms', 'CensusAge75To84','CensusAgePercent75To84', 'CensusAgePercent20To24','CensusHouseholdsPercentNonFamily', 'CensusAge25To34','CensusAgePercent18To24', 'IsInside', 'CensusGeoLevel','CensusRaceHawaiianPIOnly', 'CensusAgeOver18Female','CensusAgeOver62', 'CensusAgeOver65', 'CensusHispanicCuban','CensusAgePercent35To44', 'CensusHouseholdsPercentWithOver65','CensusRacePercentWhiteOnly', 'CensusAge18To24','CensusHispanicNonHispanic', 'CensusAgePercent18To19','CensusHouseholdsTotalFamilies', 'CensusAgeOver18Male','CensusHouseholdsSingleMoms','CensusTotalPopulationChange2000To2010', 'CensusRaceIndian','CensusRacePercentBlackOnly', 'CensusHouseholdsFamilyKidsUnder18','CensusAreaName', 'CensusHouseholdsLivingAlone','CensusAgePercent25To44', 'CensusRacePercentMultiRace','CensusRaceOtherOnly', 'CensusPeoplePerSqMi','CensusHouseholdsFemaleHouseholder', 'CensusAge5To9','CensusRacePercentOneRaceOnly', 'CensusHispanicPercentPuertoRican','CensusHispanicPercentCuban', 'CensusAgePercentOver21','CensusHispanicPercentHispanic','CensusHouseholdsPercentMarriedCoupleKidsUnder18','CensusHouseholdsMarriedCouple', 'CensusAgePercent25To34','CensusAgePercentUnder18', 'CensusHouseholdsAverageFamilySize','CensusAgePercent0To4', 'CensusHouseholdsAverageHouseholdSize','ClassCode', 'ZctaCode', 'CensusSexPercentMale', 'CensusAge35To44','CensusAgePercentOver18Male','CensusHouseholdsPercentUnmarriedPartner','CensusHispanicOtherHispanic', 'CensusAgeOver85', 'LandAreaSqM','CensusAge65To74', 'DayOfWeek_Friday', 'DayOfWeek_Monday','DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday','DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'PdDistrict_BAYVIEW','PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION','PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND','PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL','PdDistrict_TENDERLOIN', 'is_weekend', 'time_of_day_bucket','day_of_month', 'month_of_year', 'year']

X = c[['zip','CensusAgeMedianAge', 'CensusRacePercentIndian', 'CensusRacePercentAsianOnly', 'CensusTotalPopulation', 'CensusSexMale', 'CensusAgePercent15To19','CensusHispanicMexican', 'CensusSexFemale', 'CensusHouseholdsHouseholderOver65', 'CensusRaceWhiteOnly', 'CensusHouseholdsPercentSameSexPartner', 'CensusHouseholdsPercentLivingAlone','CensusHouseholdsPercentFamilies', CensusRaceBlackOnly', 'CensusAgePercentOver65','CensusAgePercentOver62', 'CensusHispanicTotalPopulation', 'CensusAgePercent65To74', 'CensusAgePercent55To59', 'CensusAgePercentOver18', 'CensusAgePercent45To64','CensusRaceAsianOnly', 'CensusHouseholdsPercentFemaleHouseholder', 'CensusAge55To59', 'CensusHouseholdsSameSexPartner', 'CensusHispanicPercentMexican', 'CensusHispanicPuertoRican', 'CensusSexPercentFemale', 'CensusHouseholdsPercentMarriedCouple','CensusHouseholdsPercentSingleMoms','CensusAgePercent75To84', 'CensusRaceHawaiianPIOnly', 'CensusAgeOver65', 'CensusHispanicCuban', 'CensusRacePercentWhiteOnly', 'CensusAge18To24', 'CensusHouseholdsSingleMoms', 'CensusRaceIndian','CensusRacePercentBlackOnly', 'CensusHouseholdsFemaleHouseholder', 'CensusRacePercentOneRaceOnly', 'CensusHispanicPercentPuertoRican','CensusHispanicPercentCuban', CensusHispanicPercentHispanic', 'CensusAgePercent25To34', 'CensusHouseholdsAverageHouseholdSize', 'DayOfWeek_Friday', 'DayOfWeek_Monday','DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday','DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'PdDistrict_BAYVIEW','PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION','PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND','PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL','PdDistrict_TENDERLOIN', 'is_weekend', 'time_of_day_bucket','day_of_month', 'month_of_year', 'year']]
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


