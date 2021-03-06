# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:52:47 2015

@author: braydencleary
"""

import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime
import re

WEEKEND_DAYS    = ['Saturday', 'Sunday']
EARLY_MORNING   = [5,6,7]
LATE_MORNING    = [8,9,10]
EARLY_AFTERNOON = [11,12,13]
LATE_AFTERNOON  = [14,15,16]
EARLY_EVENING   = [17,18,19]
LATE_EVENING    = [20,21,22]
EARLY_NIGHT     = [23,0,1]
LATE_NIGHT      = [2,3,4]

def determine_time_of_day_bucket(date):
  hour = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').hour
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

crime_train = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_train.csv')
zips        = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/zips_train.csv')
additional_data_for_zips_train = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/additional_data_for_zips_train.csv')
zips.drop_duplicates(subset='index_of_crime_train', inplace=True)
zips.set_index('index_of_crime_train', inplace=True)
crime_train = pd.merge(crime_train, zips, how='inner', left_index=True, right_index=True)
crime_train = pd.merge(crime_train, additional_data_for_zips_train, how='outer', left_on=' zip', right_on=' ZctaCode')
crime_train.rename(columns=lambda x: x.strip(), inplace=True)

dummies = pd.get_dummies(crime_train, columns=['DayOfWeek', 'PdDistrict'])
for column_name, column in dummies.transpose().iterrows():
  crime_train[column_name] = column
crime_train['is_weekend'] = crime_train.apply(lambda x: 1 if x['DayOfWeek'] in WEEKEND_DAYS else 0, axis=1)
crime_train['time_of_day_bucket'] = crime_train.apply(lambda x: determine_time_of_day_bucket(x['Dates']), axis=1)
crime_train['day_of_month'] = crime_train.apply(lambda x: datetime.strptime(x['Dates'], '%Y-%m-%d %H:%M:%S').day, axis=1)
crime_train['month_of_year'] = crime_train.apply(lambda x: datetime.strptime(x['Dates'], '%Y-%m-%d %H:%M:%S').month, axis=1)
crime_train['year'] = crime_train.apply(lambda x: datetime.strptime(x['Dates'], '%Y-%m-%d %H:%M:%S').year, axis=1)
