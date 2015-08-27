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
import csv

c = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_with_features.csv')
response = 'Category'
y = c[response]
sorted_features = ['day_of_month', 'time_of_day_bucket', 'month_of_year', 'year','DayOfWeek_Friday', 'DayOfWeek_Wednesday', 'DayOfWeek_Tuesday','DayOfWeek_Thursday', 'DayOfWeek_Monday', 'PdDistrict_TENDERLOIN','PdDistrict_NORTHERN', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday','is_weekend', 'PdDistrict_SOUTHERN', 'PdDistrict_CENTRAL','PdDistrict_PARK', 'PdDistrict_MISSION','CensusRacePercentOtherOnly', 'PdDistrict_INGLESIDE','CensusRaceOtherOnly', 'CensusHispanicPercentMexican','PdDistrict_BAYVIEW', 'CensusHispanicMexican', 'CensusAge18To19','CensusAgePercent15To19', 'PdDistrict_RICHMOND','PdDistrict_TARAVAL', 'CensusAge15To17', 'CensusAgeUnder18','CensusHispanicPercentOtherHispanic','CensusHouseholdsPercentLivingAlone','CensusHouseholdsFemaleHouseholder', 'CensusRaceIndian','CensusHouseholdsHouseholdsWithUnder18','CensusHispanicPuertoRican', 'CensusAge10To14', 'CensusAge45To54','CensusHouseholdsAverageFamilySize','CensusHouseholdsPercentMarriedCouple', 'CensusAge45To64','CensusHouseholdsPercentFamilies', 'CensusAgePercentOver18Male','CensusRaceHawaiianPIOnly', 'CensusRaceBlackOnly','CensusAge15To19', 'CensusHouseholdsAverageHouseholdSize','CensusHispanicOtherHispanic','CensusHouseholdsPercentFamilyKidsUnder18', 'ZctaCode','CensusHispanicTotalPopulation', 'CensusHispanicCuban','CensusAgePercent0To4', 'CensusAgePercentOver18','CensusHouseholdsMarriedCoupleKidsUnder18','CensusHouseholdsPercentSingleMoms', 'CensusHouseholdsSingleMoms','CensusAge55To59', 'CensusHouseholdsFamilyKidsUnder18','CensusAge5To9', 'CensusAge0To4','CensusHouseholdsPercentHouseholdsWithUnder18','CensusAgePercent5To9', 'CensusAge18To24','CensusRacePercentIndian', 'CensusHispanicPercentHispanic','CensusHouseholdsPercentMarriedCoupleKidsUnder18','CensusHispanicPercentCuban', 'CensusAgePercentUnder18','CensusAgeMedianAge', 'CensusHouseholdsLivingAlone','CensusAge20To24', 'CensusHouseholdsPercentFemaleHouseholder','CensusAgePercentOver21', 'zip', 'CensusAgePercent10To14','CensusHouseholdsTotalFamilies', 'CensusRaceMultiRace','CensusAgePercentOver18Female', 'CensusAgePercent18To19', 'GeoId','CensusHispanicPercentPuertoRican', 'CensusSexPercentMale','CensusAgePercentOver65Male', 'CensusAgePercent15To17','CensusRaceOneRaceOnly', 'CensusHispanicNonHispanic','CensusAgePercent55To59', 'CensusAgePercent45To64','CensusAge60To64', 'CensusAgePercent65To74','CensusHouseholdsPercentNonFamily', 'CensusAgeOver18Male','CensusHouseholdsWithOver65', 'CensusHouseholdsNonFamily','CensusAgeOver62', 'CensusAgePercent45To54','CensusHouseholdsMarriedCouple', 'CensusHispanicPercentNonHispanic','CensusAge65To74', 'CensusAgeOver18Female', 'LongitudeDeg','LandAreaSqM', 'LatitudeDeg', 'CensusPeoplePerSqMi', 'AskGeoId','CensusAgePercentOver85', 'CensusSexMale','CensusAgePercentOver65Female', 'CensusAgePercent20To24','CensusHouseholdsPercentUnmarriedPartner','CensusHispanicPercentWhiteNonHispanic', 'CensusAgePercentOver65','CensusAgePercent35To44', 'CensusTotalPopulation','CensusHouseholdsTotal', 'CensusRacePercentMultiRace','CensusSexPercentFemale', 'CensusAgePercent60To64','CensusAgePercentOver62', 'CensusRaceAsianOnly','CensusRacePercentAsianOnly', 'CensusHispanicWhiteNonHispanic','WaterAreaSqM', 'CensusRacePercentBlackOnly','CensusHouseholdsHouseholderOver65', 'CensusAgePercent25To34','CensusAgePercent75To84', 'CensusAgeOver21','CensusHouseholdsPercentHouseholderOver65','CensusHouseholdsUnmarriedPartner','CensusHouseholdsSameSexPartner', 'CensusAge35To44','CensusHouseholdsPercentWithOver65','CensusRacePercentHawaiianPIOnly', 'CensusSexFemale','CensusRacePercentWhiteOnly', 'CensusRaceWhiteOnly','CensusAge25To44', 'CensusRacePercentOneRaceOnly','CensusAgeOver65', 'CensusAgeOver65Male', 'CensusEsriId','CensusHouseholdsPercentSameSexPartner', 'CensusAge75To84','CensusAgeOver18', 'CensusAgePercent25To44','CensusAgePercent18To24', 'CensusAgeOver85','CensusAgeOver65Female', 'CensusAge25To34', 'CensusGeoId','MinDistanceKm', 'CensusGeoLevel']

features_to_use = sorted_features[0:5]

X = c[features_to_use]

crime_test = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_test_with_features.csv')

ctree = tree.DecisionTreeClassifier(max_depth=1)
model = ctree.fit(X, y)

features = ['day_of_month', 'time_of_day_bucket', 'month_of_year', 'year','DayOfWeek_Friday']
predicted_probs = model.predict_proba(crime_test[features])


with open("first_submission.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(predicted_probs)


for row in crime_test.iterrows():
  fd = open('predictions.csv','a')
  hey = np.array(model.predict_proba(row[1][features])[0])
  print(hey)

type(first_submission[0])
first_submission = open("/Volumes/brayden/first_submission.csv","r").readlines()
id = 0
for row in first_submission:
  fd = open('with_ids.csv','a')
  fd.write(str(id) + ',' + row)
  fd.close()
  print(id)
  id += 1


