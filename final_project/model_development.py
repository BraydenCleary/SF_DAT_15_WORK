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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, linear_model, datasets

# Read in dataset
c = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_with_features.csv')
# drop columns where zip is 0 (gps coordinates were whack)
c = c[c.zip != 0].sample(10000, random_state=1)

# columns ['Unnamed: 0', 'Dates', 'Category', 'Descript', 'DayOfWeek','PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'zip',
relevant_columns = ['zip','CensusAgeMedianAge', 'CensusAgePercent10To14', 'LatitudeDeg','CensusHouseholdsWithOver65','CensusHouseholdsPercentHouseholdsWithUnder18', 'CensusAgeOver18','CensusHouseholdsPercentFamilyKidsUnder18','CensusRacePercentIndian', 'CensusRacePercentAsianOnly','CensusAgeUnder18', 'CensusTotalPopulation','CensusAgePercent45To54', 'CensusAgePercentOver65Female','CensusGeoId', 'CensusAgePercentOver18Female','CensusRacePercentHawaiianPIOnly','CensusHouseholdsPercentHouseholderOver65', 'CensusAge18To19','CensusSexMale', 'CensusEsriId', 'CensusAgeOver21','CensusAgePercent15To17', 'CensusAgePercent15To19','CensusHispanicMexican', 'CensusHispanicPercentNonHispanic','CensusAge0To4', 'CensusSexFemale', 'LongitudeDeg','CensusAgePercent5To9', 'CensusAge20To24','CensusHouseholdsHouseholderOver65', 'CensusRaceWhiteOnly','CensusAge60To64', 'CensusHouseholdsPercentSameSexPartner','CensusAge25To44', 'CensusRacePercentOtherOnly', 'MinDistanceKm','CensusHouseholdsPercentLivingAlone','CensusHouseholdsPercentFamilies', 'FeatureClassCode','CensusHouseholdsUnmarriedPartner', 'CensusHouseholdsNonFamily','CensusRaceBlackOnly', 'WaterAreaSqM', 'CensusAgePercentOver65','CensusAgePercentOver62', 'FunctionalStatus','CensusHispanicTotalPopulation', 'CensusHispanicWhiteNonHispanic','CensusHispanicPercentOtherHispanic','CensusTotalPopulationPercentChange2000To2010','CensusRaceOneRaceOnly', 'CensusAgePercent65To74','CensusAge45To54', 'CensusAgePercent55To59', 'AskGeoId','CensusAge15To19', 'CensusAgeOver65Male', 'CensusAge15To17','CensusAgePercentOver18', 'CensusAgePercentOver65Male','CensusHouseholdsTotal', 'CensusAgePercent45To64','CensusRaceAsianOnly', 'CensusAgeOver65Female', 'GeoId','CensusAge45To64', 'CensusGeoCode', 'CensusStateAbbreviation','CensusAgePercentOver85', 'CensusAge10To14','CensusHouseholdsPercentFemaleHouseholder', 'CensusAge55To59','CensusHouseholdsSameSexPartner', 'CensusRaceMultiRace','CensusHispanicPercentMexican', 'CensusHispanicPuertoRican','CensusAgePercent60To64', 'CensusSexPercentFemale','CensusHouseholdsMarriedCoupleKidsUnder18','CensusHispanicPercentWhiteNonHispanic','CensusTotalPopulationIn2000','CensusHouseholdsPercentMarriedCouple','CensusHouseholdsHouseholdsWithUnder18','CensusHouseholdsPercentSingleMoms', 'CensusAge75To84','CensusAgePercent75To84', 'CensusAgePercent20To24','CensusHouseholdsPercentNonFamily', 'CensusAge25To34','CensusAgePercent18To24', 'IsInside', 'CensusGeoLevel','CensusRaceHawaiianPIOnly', 'CensusAgeOver18Female','CensusAgeOver62', 'CensusAgeOver65', 'CensusHispanicCuban','CensusAgePercent35To44', 'CensusHouseholdsPercentWithOver65','CensusRacePercentWhiteOnly', 'CensusAge18To24','CensusHispanicNonHispanic', 'CensusAgePercent18To19','CensusHouseholdsTotalFamilies', 'CensusAgeOver18Male','CensusHouseholdsSingleMoms','CensusTotalPopulationChange2000To2010', 'CensusRaceIndian','CensusRacePercentBlackOnly', 'CensusHouseholdsFamilyKidsUnder18','CensusAreaName', 'CensusHouseholdsLivingAlone','CensusAgePercent25To44', 'CensusRacePercentMultiRace','CensusRaceOtherOnly', 'CensusPeoplePerSqMi','CensusHouseholdsFemaleHouseholder', 'CensusAge5To9','CensusRacePercentOneRaceOnly', 'CensusHispanicPercentPuertoRican','CensusHispanicPercentCuban', 'CensusAgePercentOver21','CensusHispanicPercentHispanic','CensusHouseholdsPercentMarriedCoupleKidsUnder18','CensusHouseholdsMarriedCouple', 'CensusAgePercent25To34','CensusAgePercentUnder18', 'CensusHouseholdsAverageFamilySize','CensusAgePercent0To4', 'CensusHouseholdsAverageHouseholdSize','ClassCode', 'ZctaCode', 'CensusSexPercentMale', 'CensusAge35To44','CensusAgePercentOver18Male','CensusHouseholdsPercentUnmarriedPartner','CensusHispanicOtherHispanic', 'CensusAgeOver85', 'LandAreaSqM','CensusAge65To74', 'DayOfWeek_Friday', 'DayOfWeek_Monday','DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday','DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'PdDistrict_BAYVIEW','PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION','PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND','PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL','PdDistrict_TENDERLOIN', 'is_weekend', 'time_of_day_bucket','day_of_month', 'month_of_year', 'year']
# ,
 #
 # ,
 #
 #'CensusHouseholdsNonFamily','CensusRaceBlackOnly', 'WaterAreaSqM', 'CensusAgePercentOver65','CensusAgePercentOver62', 'FunctionalStatus','CensusHispanicTotalPopulation', 'CensusHispanicWhiteNonHispanic','CensusHispanicPercentOtherHispanic','CensusTotalPopulationPercentChange2000To2010','CensusRaceOneRaceOnly', 'CensusAgePercentOver18', 'CensusAgePercentOver65Male','CensusHouseholdsTotal', 'CensusAgePercent45To64','CensusRaceAsianOnly', 'CensusAgeOver65Female', 'GeoId','CensusAge45To64', 'CensusGeoCode', 'CensusStateAbbreviation','CensusAgePercentOver85', 'CensusAge10To14','CensusHouseholdsPercentFemaleHouseholder', 'CensusAge55To59','CensusHouseholdsSameSexPartner', 'CensusRaceMultiRace','CensusHispanicPercentMexican', 'CensusHispanicPuertoRican','CensusAgePercent60To64', 'CensusSexPercentFemale','CensusHouseholdsMarriedCoupleKidsUnder18','CensusHispanicPercentWhiteNonHispanic','CensusTotalPopulationIn2000','CensusHouseholdsPercentMarriedCouple','CensusHouseholdsHouseholdsWithUnder18','CensusHouseholdsPercentSingleMoms', 'CensusAge75To84','CensusAgePercent75To84', 'CensusAgePercent20To24','CensusHouseholdsPercentNonFamily', 'CensusAge25To34','CensusAgePercent18To24', 'IsInside', 'CensusGeoLevel','CensusRaceHawaiianPIOnly', 'CensusAgeOver18Female','CensusAgeOver62', 'CensusAgeOver65', 'CensusHispanicCuban','CensusAgePercent35To44', 'CensusHouseholdsPercentWithOver65','CensusRacePercentWhiteOnly', 'CensusAge18To24','CensusHispanicNonHispanic', 'CensusAgePercent18To19','CensusHouseholdsTotalFamilies', 'CensusAgeOver18Male','CensusHouseholdsSingleMoms','CensusTotalPopulationChange2000To2010', 'CensusRaceIndian','CensusRacePercentBlackOnly', 'CensusHouseholdsFamilyKidsUnder18','CensusAreaName', 'CensusHouseholdsLivingAlone','CensusAgePercent25To44', 'CensusRacePercentMultiRace','CensusRaceOtherOnly', 'CensusPeoplePerSqMi','CensusHouseholdsFemaleHouseholder', 'CensusAge5To9','CensusRacePercentOneRaceOnly', 'CensusHispanicPercentPuertoRican','CensusHispanicPercentCuban', 'CensusAgePercentOver21','CensusHispanicPercentHispanic','CensusHouseholdsPercentMarriedCoupleKidsUnder18','CensusHouseholdsMarriedCouple', 'CensusAgePercent25To34','CensusAgePercentUnder18', 'CensusHouseholdsAverageFamilySize','CensusAgePercent0To4', 'CensusHouseholdsAverageHouseholdSize','ClassCode', 'ZctaCode', 'CensusSexPercentMale', 'CensusAge35To44','CensusAgePercentOver18Male','CensusHouseholdsPercentUnmarriedPartner','CensusHispanicOtherHispanic', 'CensusAgeOver85', 'LandAreaSqM','CensusAge65To74', 'CensusRacePercentIndian', 'CensusRacePercentAsianOnly', 'CensusTotalPopulation', 'CensusSexMale', 'CensusAgePercent15To19','CensusHispanicMexican', 'CensusSexFemale', 'CensusHouseholdsHouseholderOver65', 'CensusRaceWhiteOnly', 'CensusHouseholdsPercentSameSexPartner', 'CensusHouseholdsPercentLivingAlone','CensusHouseholdsPercentFamilies', 'CensusRaceBlackOnly', 'CensusAgePercentOver65','CensusAgePercentOver62', 'CensusHispanicTotalPopulation', 'CensusAgePercent65To74', 'CensusAgePercent55To59', 'CensusAgePercentOver18', 'CensusAgePercent45To64','CensusRaceAsianOnly', 'CensusHouseholdsPercentFemaleHouseholder', 'CensusAge55To59', 'CensusHouseholdsSameSexPartner', 'CensusHispanicPercentMexican', 'CensusHispanicPuertoRican', 'CensusSexPercentFemale', 'CensusHouseholdsPercentMarriedCouple','CensusHouseholdsPercentSingleMoms','CensusAgePercent75To84', 'CensusRaceHawaiianPIOnly', 'CensusAgeOver65', 'CensusHispanicCuban', 'CensusRacePercentWhiteOnly', 'CensusAge18To24', 'CensusHouseholdsSingleMoms', ]
features = list(set(['zip','CensusAgeMedianAge', 'is_weekend', 'time_of_day_bucket', 'year', 'PdDistrict_TENDERLOIN', 'PdDistrict_MISSION', 'CensusHouseholdsPercentSingleMoms', 'CensusAgeMedianAge', 'CensusAgePercent10To14', 'LatitudeDeg','CensusHouseholdsWithOver65','CensusHouseholdsPercentHouseholdsWithUnder18', 'CensusAgeOver18','CensusHouseholdsPercentFamilyKidsUnder18','CensusRacePercentIndian', 'CensusRacePercentAsianOnly','CensusAgeUnder18', 'CensusTotalPopulation','CensusAgePercent45To54', 'CensusAgePercentOver65Female','CensusGeoId', 'CensusAgePercentOver18Female','CensusRacePercentHawaiianPIOnly','CensusHouseholdsPercentHouseholderOver65', 'CensusAge18To19','CensusSexMale', 'CensusEsriId', 'CensusAgeOver21','CensusAgePercent15To17', 'CensusAgePercent15To19','CensusHispanicMexican', 'CensusHispanicPercentNonHispanic','CensusAge0To4', 'CensusSexFemale', 'LongitudeDeg','CensusAgePercent5To9', 'CensusAge20To24','CensusHouseholdsHouseholderOver65', 'CensusRaceWhiteOnly','CensusAge60To64', 'CensusRaceIndian', 'CensusHouseholdsFemaleHouseholder', 'CensusRacePercentOneRaceOnly', 'CensusHispanicPercentPuertoRican','CensusHispanicPercentCuban', 'CensusHispanicPercentHispanic', 'CensusAgePercent25To34', 'CensusHouseholdsAverageHouseholdSize', 'PdDistrict_BAYVIEW','PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND','PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL','day_of_month', 'month_of_year','DayOfWeek_Friday', 'DayOfWeek_Monday','DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday','DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'CensusRacePercentBlackOnly', 'DayOfWeek_Friday', 'DayOfWeek_Monday','DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday','DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'PdDistrict_BAYVIEW','PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION','PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND','PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL','PdDistrict_TENDERLOIN', 'is_weekend', 'time_of_day_bucket','day_of_month', 'month_of_year', 'year', 'CensusAgePercent65To74','CensusAge45To54', 'CensusAgePercent55To59', 'AskGeoId','CensusAge15To19', 'CensusAgeOver65Male', 'CensusAge15To17', 'CensusHouseholdsPercentSameSexPartner','CensusAge25To44', 'CensusRacePercentOtherOnly', 'MinDistanceKm','CensusHouseholdsPercentLivingAlone','CensusHouseholdsPercentFamilies', 'FeatureClassCode','CensusHouseholdsUnmarriedPartner']))
response = 'Category'
X = c[features]
y = c[response]

# Use grid search
knn = KNeighborsClassifier()
k_range = range(1, 30, 1)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
grid.best_score_
grid.grid_scores_
grid.best_params_

# most popular category of crime is Larceny/Theft which occured 174,885 times out of 877982 crimes...so my
# baseline for model effectiveness is 19.92%

# knn where k is 23 and features are 'zip','CensusAgeMedianAge', 'is_weekend', 'time_of_day_bucket', 'year', 'PdDistrict_TENDERLOIN', 'PdDistrict_MISSION', 'CensusHouseholdsPercentSingleMoms'
# is about 21% accurate

####

c = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_with_features.csv')

c = c[c.zip != 0]

features = ['is_weekend', 'PdDistrict_TENDERLOIN', 'PdDistrict_MISSION','DayOfWeek_Friday', 'DayOfWeek_Monday','DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday','DayOfWeek_Tuesday', 'DayOfWeek_Wednesday','PdDistrict_BAYVIEW','PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', ,'PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND','PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL']
response = 'Category'
X = c[features]
y = c[response]

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

# 8/15
##### Random forest to determine important features #####


c = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_with_features.csv')
# drop columns where zip is 0 (gps coordinates were whack)
small = c[c.zip != 0].sample(10000, random_state=1)
c = c[c.zip != 0]

response = 'Category'
y_small = small[response]
y = c[response]

# columns ['Unnamed: 0', 'Dates', 'Category', 'Descript', 'DayOfWeek','PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'zip',
relevant_features = ['zip','CensusAgeMedianAge', 'CensusAgePercent10To14', 'LatitudeDeg','CensusHouseholdsWithOver65','CensusHouseholdsPercentHouseholdsWithUnder18', 'CensusAgeOver18','CensusHouseholdsPercentFamilyKidsUnder18','CensusRacePercentIndian', 'CensusRacePercentAsianOnly','CensusAgeUnder18', 'CensusTotalPopulation','CensusAgePercent45To54', 'CensusAgePercentOver65Female','CensusGeoId', 'CensusAgePercentOver18Female','CensusRacePercentHawaiianPIOnly','CensusHouseholdsPercentHouseholderOver65', 'CensusAge18To19','CensusSexMale', 'CensusEsriId', 'CensusAgeOver21','CensusAgePercent15To17', 'CensusAgePercent15To19','CensusHispanicMexican', 'CensusHispanicPercentNonHispanic','CensusAge0To4', 'CensusSexFemale', 'LongitudeDeg','CensusAgePercent5To9', 'CensusAge20To24','CensusHouseholdsHouseholderOver65', 'CensusRaceWhiteOnly','CensusAge60To64', 'CensusHouseholdsPercentSameSexPartner','CensusAge25To44', 'CensusRacePercentOtherOnly', 'MinDistanceKm','CensusHouseholdsPercentLivingAlone','CensusHouseholdsPercentFamilies', 'FeatureClassCode','CensusHouseholdsUnmarriedPartner', 'CensusHouseholdsNonFamily','CensusRaceBlackOnly', 'WaterAreaSqM', 'CensusAgePercentOver65','CensusAgePercentOver62', 'FunctionalStatus','CensusHispanicTotalPopulation', 'CensusHispanicWhiteNonHispanic','CensusHispanicPercentOtherHispanic','CensusTotalPopulationPercentChange2000To2010','CensusRaceOneRaceOnly', 'CensusAgePercent65To74','CensusAge45To54', 'CensusAgePercent55To59', 'AskGeoId','CensusAge15To19', 'CensusAgeOver65Male', 'CensusAge15To17','CensusAgePercentOver18', 'CensusAgePercentOver65Male','CensusHouseholdsTotal', 'CensusAgePercent45To64','CensusRaceAsianOnly', 'CensusAgeOver65Female', 'GeoId','CensusAge45To64', 'CensusGeoCode', 'CensusStateAbbreviation','CensusAgePercentOver85', 'CensusAge10To14','CensusHouseholdsPercentFemaleHouseholder', 'CensusAge55To59','CensusHouseholdsSameSexPartner', 'CensusRaceMultiRace','CensusHispanicPercentMexican', 'CensusHispanicPuertoRican','CensusAgePercent60To64', 'CensusSexPercentFemale','CensusHouseholdsMarriedCoupleKidsUnder18','CensusHispanicPercentWhiteNonHispanic','CensusTotalPopulationIn2000','CensusHouseholdsPercentMarriedCouple','CensusHouseholdsHouseholdsWithUnder18','CensusHouseholdsPercentSingleMoms', 'CensusAge75To84','CensusAgePercent75To84', 'CensusAgePercent20To24','CensusHouseholdsPercentNonFamily', 'CensusAge25To34','CensusAgePercent18To24', 'IsInside', 'CensusGeoLevel','CensusRaceHawaiianPIOnly', 'CensusAgeOver18Female','CensusAgeOver62', 'CensusAgeOver65', 'CensusHispanicCuban','CensusAgePercent35To44', 'CensusHouseholdsPercentWithOver65','CensusRacePercentWhiteOnly', 'CensusAge18To24','CensusHispanicNonHispanic', 'CensusAgePercent18To19','CensusHouseholdsTotalFamilies', 'CensusAgeOver18Male','CensusHouseholdsSingleMoms','CensusTotalPopulationChange2000To2010', 'CensusRaceIndian','CensusRacePercentBlackOnly', 'CensusHouseholdsFamilyKidsUnder18','CensusAreaName', 'CensusHouseholdsLivingAlone','CensusAgePercent25To44', 'CensusRacePercentMultiRace','CensusRaceOtherOnly', 'CensusPeoplePerSqMi','CensusHouseholdsFemaleHouseholder', 'CensusAge5To9','CensusRacePercentOneRaceOnly', 'CensusHispanicPercentPuertoRican','CensusHispanicPercentCuban', 'CensusAgePercentOver21','CensusHispanicPercentHispanic','CensusHouseholdsPercentMarriedCoupleKidsUnder18','CensusHouseholdsMarriedCouple', 'CensusAgePercent25To34','CensusAgePercentUnder18', 'CensusHouseholdsAverageFamilySize','CensusAgePercent0To4', 'CensusHouseholdsAverageHouseholdSize','ClassCode', 'ZctaCode', 'CensusSexPercentMale', 'CensusAge35To44','CensusAgePercentOver18Male','CensusHouseholdsPercentUnmarriedPartner','CensusHispanicOtherHispanic', 'CensusAgeOver85', 'LandAreaSqM','CensusAge65To74', 'DayOfWeek_Friday', 'DayOfWeek_Monday','DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday','DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'PdDistrict_BAYVIEW','PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION','PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND','PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL','PdDistrict_TENDERLOIN', 'is_weekend', 'time_of_day_bucket','day_of_month', 'month_of_year', 'year']

good_features = []
for feature in relevant_features:
  try:
    X_small = pd.DataFrame(small[feature])
    rfclf = RandomForestClassifier(n_estimators=1, max_features='auto', oob_score=True)
    rfclf.fit(X_small, y_small)
    good_features.append(feature)
  except:
    pass

good_features = ['zip', 'CensusAgeMedianAge', 'CensusAgePercent10To14', 'LatitudeDeg', 'CensusHouseholdsWithOver65', 'CensusHouseholdsPercentHouseholdsWithUnder18', 'CensusAgeOver18', 'CensusHouseholdsPercentFamilyKidsUnder18', 'CensusRacePercentIndian', 'CensusRacePercentAsianOnly', 'CensusAgeUnder18', 'CensusTotalPopulation', 'CensusAgePercent45To54', 'CensusAgePercentOver65Female', 'CensusGeoId', 'CensusAgePercentOver18Female', 'CensusRacePercentHawaiianPIOnly', 'CensusHouseholdsPercentHouseholderOver65', 'CensusAge18To19', 'CensusSexMale', 'CensusEsriId', 'CensusAgeOver21', 'CensusAgePercent15To17', 'CensusAgePercent15To19', 'CensusHispanicMexican', 'CensusHispanicPercentNonHispanic', 'CensusAge0To4', 'CensusSexFemale', 'LongitudeDeg', 'CensusAgePercent5To9', 'CensusAge20To24', 'CensusHouseholdsHouseholderOver65', 'CensusRaceWhiteOnly', 'CensusAge60To64', 'CensusHouseholdsPercentSameSexPartner', 'CensusAge25To44', 'CensusRacePercentOtherOnly', 'MinDistanceKm', 'CensusHouseholdsPercentLivingAlone', 'CensusHouseholdsPercentFamilies', 'CensusHouseholdsUnmarriedPartner', 'CensusHouseholdsNonFamily', 'CensusRaceBlackOnly', 'WaterAreaSqM', 'CensusAgePercentOver65', 'CensusAgePercentOver62', 'CensusHispanicTotalPopulation', 'CensusHispanicWhiteNonHispanic', 'CensusHispanicPercentOtherHispanic', 'CensusRaceOneRaceOnly', 'CensusAgePercent65To74', 'CensusAge45To54', 'CensusAgePercent55To59', 'AskGeoId', 'CensusAge15To19', 'CensusAgeOver65Male', 'CensusAge15To17', 'CensusAgePercentOver18', 'CensusAgePercentOver65Male', 'CensusHouseholdsTotal', 'CensusAgePercent45To64', 'CensusRaceAsianOnly', 'CensusAgeOver65Female', 'GeoId', 'CensusAge45To64', 'CensusAgePercentOver85', 'CensusAge10To14', 'CensusHouseholdsPercentFemaleHouseholder', 'CensusAge55To59', 'CensusHouseholdsSameSexPartner', 'CensusRaceMultiRace', 'CensusHispanicPercentMexican', 'CensusHispanicPuertoRican', 'CensusAgePercent60To64', 'CensusSexPercentFemale', 'CensusHouseholdsMarriedCoupleKidsUnder18', 'CensusHispanicPercentWhiteNonHispanic', 'CensusHouseholdsPercentMarriedCouple', 'CensusHouseholdsHouseholdsWithUnder18', 'CensusHouseholdsPercentSingleMoms', 'CensusAge75To84', 'CensusAgePercent75To84', 'CensusAgePercent20To24', 'CensusHouseholdsPercentNonFamily', 'CensusAge25To34', 'CensusAgePercent18To24', 'CensusGeoLevel', 'CensusRaceHawaiianPIOnly', 'CensusAgeOver18Female', 'CensusAgeOver62', 'CensusAgeOver65', 'CensusHispanicCuban', 'CensusAgePercent35To44', 'CensusHouseholdsPercentWithOver65', 'CensusRacePercentWhiteOnly', 'CensusAge18To24', 'CensusHispanicNonHispanic', 'CensusAgePercent18To19', 'CensusHouseholdsTotalFamilies', 'CensusAgeOver18Male', 'CensusHouseholdsSingleMoms', 'CensusRaceIndian', 'CensusRacePercentBlackOnly', 'CensusHouseholdsFamilyKidsUnder18', 'CensusHouseholdsLivingAlone', 'CensusAgePercent25To44', 'CensusRacePercentMultiRace', 'CensusRaceOtherOnly', 'CensusPeoplePerSqMi', 'CensusHouseholdsFemaleHouseholder', 'CensusAge5To9', 'CensusRacePercentOneRaceOnly', 'CensusHispanicPercentPuertoRican', 'CensusHispanicPercentCuban', 'CensusAgePercentOver21', 'CensusHispanicPercentHispanic', 'CensusHouseholdsPercentMarriedCoupleKidsUnder18', 'CensusHouseholdsMarriedCouple', 'CensusAgePercent25To34', 'CensusAgePercentUnder18', 'CensusHouseholdsAverageFamilySize', 'CensusAgePercent0To4', 'CensusHouseholdsAverageHouseholdSize', 'ZctaCode', 'CensusSexPercentMale', 'CensusAge35To44', 'CensusAgePercentOver18Male', 'CensusHouseholdsPercentUnmarriedPartner', 'CensusHispanicOtherHispanic', 'CensusAgeOver85', 'LandAreaSqM', 'CensusAge65To74', 'DayOfWeek_Friday', 'DayOfWeek_Monday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE', 'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK', 'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN', 'is_weekend', 'time_of_day_bucket', 'day_of_month', 'month_of_year', 'year']

rfclf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1)
rfclf.fit(c[good_features], y)
sorted_features = np.array(pd.DataFrame({'feature': good_features, 'importance': rfclf.feature_importances_ }).sort(columns='importance', ascending=False)['feature'])

## result of most important features according to random forest
sorted_features = ['day_of_month', 'time_of_day_bucket', 'month_of_year', 'year','DayOfWeek_Friday', 'DayOfWeek_Wednesday', 'DayOfWeek_Tuesday','DayOfWeek_Thursday', 'DayOfWeek_Monday', 'PdDistrict_TENDERLOIN','PdDistrict_NORTHERN', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday','is_weekend', 'PdDistrict_SOUTHERN', 'PdDistrict_CENTRAL','PdDistrict_PARK', 'PdDistrict_MISSION','CensusRacePercentOtherOnly', 'PdDistrict_INGLESIDE','CensusRaceOtherOnly', 'CensusHispanicPercentMexican','PdDistrict_BAYVIEW', 'CensusHispanicMexican', 'CensusAge18To19','CensusAgePercent15To19', 'PdDistrict_RICHMOND','PdDistrict_TARAVAL', 'CensusAge15To17', 'CensusAgeUnder18','CensusHispanicPercentOtherHispanic','CensusHouseholdsPercentLivingAlone','CensusHouseholdsFemaleHouseholder', 'CensusRaceIndian','CensusHouseholdsHouseholdsWithUnder18','CensusHispanicPuertoRican', 'CensusAge10To14', 'CensusAge45To54','CensusHouseholdsAverageFamilySize','CensusHouseholdsPercentMarriedCouple', 'CensusAge45To64','CensusHouseholdsPercentFamilies', 'CensusAgePercentOver18Male','CensusRaceHawaiianPIOnly', 'CensusRaceBlackOnly','CensusAge15To19', 'CensusHouseholdsAverageHouseholdSize','CensusHispanicOtherHispanic','CensusHouseholdsPercentFamilyKidsUnder18', 'ZctaCode','CensusHispanicTotalPopulation', 'CensusHispanicCuban','CensusAgePercent0To4', 'CensusAgePercentOver18','CensusHouseholdsMarriedCoupleKidsUnder18','CensusHouseholdsPercentSingleMoms', 'CensusHouseholdsSingleMoms','CensusAge55To59', 'CensusHouseholdsFamilyKidsUnder18','CensusAge5To9', 'CensusAge0To4','CensusHouseholdsPercentHouseholdsWithUnder18','CensusAgePercent5To9', 'CensusAge18To24','CensusRacePercentIndian', 'CensusHispanicPercentHispanic','CensusHouseholdsPercentMarriedCoupleKidsUnder18','CensusHispanicPercentCuban', 'CensusAgePercentUnder18','CensusAgeMedianAge', 'CensusHouseholdsLivingAlone','CensusAge20To24', 'CensusHouseholdsPercentFemaleHouseholder','CensusAgePercentOver21', 'zip', 'CensusAgePercent10To14','CensusHouseholdsTotalFamilies', 'CensusRaceMultiRace','CensusAgePercentOver18Female', 'CensusAgePercent18To19', 'GeoId','CensusHispanicPercentPuertoRican', 'CensusSexPercentMale','CensusAgePercentOver65Male', 'CensusAgePercent15To17','CensusRaceOneRaceOnly', 'CensusHispanicNonHispanic','CensusAgePercent55To59', 'CensusAgePercent45To64','CensusAge60To64', 'CensusAgePercent65To74','CensusHouseholdsPercentNonFamily', 'CensusAgeOver18Male','CensusHouseholdsWithOver65', 'CensusHouseholdsNonFamily','CensusAgeOver62', 'CensusAgePercent45To54','CensusHouseholdsMarriedCouple', 'CensusHispanicPercentNonHispanic','CensusAge65To74', 'CensusAgeOver18Female', 'LongitudeDeg','LandAreaSqM', 'LatitudeDeg', 'CensusPeoplePerSqMi', 'AskGeoId','CensusAgePercentOver85', 'CensusSexMale','CensusAgePercentOver65Female', 'CensusAgePercent20To24','CensusHouseholdsPercentUnmarriedPartner','CensusHispanicPercentWhiteNonHispanic', 'CensusAgePercentOver65','CensusAgePercent35To44', 'CensusTotalPopulation','CensusHouseholdsTotal', 'CensusRacePercentMultiRace','CensusSexPercentFemale', 'CensusAgePercent60To64','CensusAgePercentOver62', 'CensusRaceAsianOnly','CensusRacePercentAsianOnly', 'CensusHispanicWhiteNonHispanic','WaterAreaSqM', 'CensusRacePercentBlackOnly','CensusHouseholdsHouseholderOver65', 'CensusAgePercent25To34','CensusAgePercent75To84', 'CensusAgeOver21','CensusHouseholdsPercentHouseholderOver65','CensusHouseholdsUnmarriedPartner','CensusHouseholdsSameSexPartner', 'CensusAge35To44','CensusHouseholdsPercentWithOver65','CensusRacePercentHawaiianPIOnly', 'CensusSexFemale','CensusRacePercentWhiteOnly', 'CensusRaceWhiteOnly','CensusAge25To44', 'CensusRacePercentOneRaceOnly','CensusAgeOver65', 'CensusAgeOver65Male', 'CensusEsriId','CensusHouseholdsPercentSameSexPartner', 'CensusAge75To84','CensusAgeOver18', 'CensusAgePercent25To44','CensusAgePercent18To24', 'CensusAgeOver85','CensusAgeOver65Female', 'CensusAge25To34', 'CensusGeoId','MinDistanceKm', 'CensusGeoLevel']
##

features_to_use = sorted_features[0:5]

X = c[features_to_use]
X_small = small[features_to_use]
y_small = small[response]

#### normalize data before running knn ####

normalized_x_small = (X_small - X_small.mean()) / X_small.std()
knn = KNeighborsClassifier()
k_range = range(1, 30, 1)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(normalized_x_small, y_small)
grid.best_score_

#  did worse with normalizing data....sheeeiiiiiiit

#### end normalize ###

knn = KNeighborsClassifier()
k_range = range(1, 30, 1)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_small, y_small)
grid.best_score_
grid.best_params_
#20% with k as 27

ctree = tree.DecisionTreeClassifier()
depth_range = range(1, 50)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5)
grid.fit(X_small, y_small)
grid.best_score_
grid.best_params_
#23.51% with max depth as 1

nb = MultinomialNB()
features_train, features_test, response_train, response_test = train_test_split(X, y)
nb.fit(features_train, response_train)
preds = nb.predict(features_test)
metrics.accuracy_score(response_test, preds)
# .5%

rfclf = RandomForestClassifier()
estimator_range = range(1, 500, 100)
param_grid = dict(n_estimators=estimator_range)
grid = GridSearchCV(rfclf, param_grid, cv=5, scoring='accuracy')
grid.fit(X_small, y_small)
# 12% with {'n_estimators': 301}

crime_categories = ['NON-CRIMINAL', 'LARCENY/THEFT', 'ARSON', 'ROBBERY', 'VEHICLE THEFT', 'OTHER OFFENSES', 'MISSING PERSON', 'WARRANTS', 'VANDALISM', 'BAD CHECKS', 'ASSAULT', 'SUSPICIOUS OCC', 'DRUG/NARCOTIC', 'SEX OFFENSES FORCIBLE', 'BURGLARY', 'TRESPASS', 'RUNAWAY', 'FORGERY/COUNTERFEITING', 'FRAUD', 'SECONDARY CODES', 'DRUNKENNESS', 'RECOVERED VEHICLE', 'WEAPON LAWS', 'PROSTITUTION', 'DRIVING UNDER THE INFLUENCE', 'LOITERING', 'LIQUOR LAWS', 'KIDNAPPING', 'DISORDERLY CONDUCT', 'SUICIDE', 'STOLEN PROPERTY', 'FAMILY OFFENSES', 'SEX OFFENSES NON FORCIBLE', 'EMBEZZLEMENT', 'EXTORTION', 'BRIBERY', 'GAMBLING']

red_wagon_disadvantageous_responses = ['VEHICLE THEFT', 'RECOVERED VEHICLE', 'STOLEN PROPERTY', 'ROBBERY', 'DRIVING UNDER THE INFLUENCE']
red_wagon_advantageous_responses    = ['ARSON', 'DRUNKENNESS', 'DRUG/NARCOTIC', 'GAMBLING',  'LOITERING']
red_wagon_columns                   = red_wagon_disadvantageous_responses + red_wagon_advantageous_responses

# safe: given a time of day, month of year, day of month, day of week, target demographic in terms of age, predict probabilities of red wagon safe responses for each zip, sum, and take zip with max sum
# safe: given a time of day, month of year, day of month, day of week, predict probabilities of red wagon dangerous responses for each zip, sum, and take zip with min sum

# take in a python datetime and zip code to predict smoothie advantageous/not advantageous
# take in a python datetime and zip code to predict smoothie disadvatageous/not disadvantageous

# need to make advantagous/disadvantageous columns

# c['red_wagon_advantageous'] = c.apply(lambda x: 1 if x['Category'] in red_wagon_advantageous_responses else 0, axis=1)
# c['red_wagon_disadvantageous'] = c.apply(lambda x: 1 if x['Category'] in red_wagon_disadvantageous_responses else 0, axis=1)

smoothies = c[c['Category'].isin(red_wagon_columns)]

smoothie_features = ['day_of_month', 'time_of_day_bucket', 'month_of_year', 'year','DayOfWeek_Friday', 'DayOfWeek_Wednesday', 'DayOfWeek_Tuesday','DayOfWeek_Thursday', 'DayOfWeek_Monday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday','is_weekend', 'zip']

smoothies['smoothie_selling_safety'] = smoothies.apply(lambda x: -1 if x['Category'] in red_wagon_disadvantageous_responses else 1, axis=1)
smoothies['smoothie_selling_safety'].value_counts()

small = smoothies.sample(10000)

X_small = small[smoothie_features]
y       = small['smoothie_selling_safety']

# decision tree (best)

ctree = tree.DecisionTreeClassifier()
depth_range = range(1, 50)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5)
grid.fit(X_small, y)
smoothie_tree = grid.best_estimator_

# logistic regression (3rd)

logreg = LogisticRegression()
grid = GridSearchCV(logreg, {}, cv=5)
grid.fit(X_small, y)
smoothie_log_reg = grid.best_estimator_

# random forest (2nd)

from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1)
grid = GridSearchCV(rfclf, {}, cv=5)
grid.fit(X_small, y)

# knn (68%)

knn = KNeighborsClassifier()
k_range = range(1, 30, 1)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_small, y)

# naive bayes (58%)

nb = MultinomialNB()
grid = GridSearchCV(nb, {}, cv=5)
grid.fit(X_small, y)

# SVM

clf = svm.SVC()
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [.000001, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 10000], 'C': [float(x) for x in range(1,50)], 'degree': range(1,20)}
grid = GridSearchCV(clf, param_grid , cv=5)
grid.fit(X_small, y)

smoothie_tree.predict(c[smoothie_features])

tree.predict_proba(np.array(c.ix[1][smoothie_features]))

# write a program that takes in a time and returns the advantageous areas (zips) of SF and the disadvantageous areas (zips)

SAN_FRANCISCO_ZIP_CODES = [94102, 94109, 94123, 94117, 94134, 94112, 94124, 94121, 94133, 94116, 94115, 94110, 94127, 94114, 94107, 94132, 94122, 94103, 94105, 94104, 94108, 94118, 94158, 94111, 94131, 94130, 94014, 94129, 94015]
# [94014, 94015] not present in sf zip codes I parsed for application


time = datetime.now()

WEEKEND_DAYS    = ['Saturday', 'Sunday']
EARLY_MORNING   = [5,6,7]
LATE_MORNING    = [8,9,10]
EARLY_AFTERNOON = [11,12,13]
LATE_AFTERNOON  = [14,15,16]
EARLY_EVENING   = [17,18,19]
LATE_EVENING    = [20,21,22]
EARLY_NIGHT     = [23,0,1]
LATE_NIGHT      = [2,3,4]

def determine_time_of_day_bucket(datetime):
  hour = datetime.hour
  if hour in EARLY_MORNING:
      return 1
  elif hour in LATE_MORNING:
      return 2
  elif hour in EARLY_AFTERNOON:
      return 3
  elif hour in LATE_AFTERNOON:
      return 4
  elif hour in EARLY_EVENING:
      return 5
  elif hour in LATE_EVENING:
      return 6
  elif hour in EARLY_NIGHT:
      return 7
  elif hour in LATE_NIGHT:
      return 8


def parse_for_map(datetime):
  zip_predictions = []
  for zip in SAN_FRANCISCO_ZIP_CODES:
    day_of_month        = datetime.day
    time_of_day_bucket  = determine_time_of_day_bucket(datetime)
    month_of_year       = datetime.month
    year                = datetime.year
    DayOfWeek_Friday    = datetime.isoweekday() == 5
    DayOfWeek_Wednesday = datetime.isoweekday() == 3
    DayOfWeek_Tuesday   = datetime.isoweekday() == 2
    DayOfWeek_Thursday  = datetime.isoweekday() == 4
    DayOfWeek_Monday    = datetime.isoweekday() == 1
    DayOfWeek_Saturday  = datetime.isoweekday() == 6
    DayOfWeek_Sunday    = datetime.isoweekday() == 7
    is_weekend          = datetime.isoweekday() in [6,7]
    zip_predictions.append({zip: smoothie_tree.predict([day_of_month, time_of_day_bucket, month_of_year, year, DayOfWeek_Friday, DayOfWeek_Wednesday, DayOfWeek_Tuesday, DayOfWeek_Thursday, DayOfWeek_Monday, DayOfWeek_Saturday, DayOfWeek_Sunday, is_weekend, zip])[0]})
  print(zip_predictions)
