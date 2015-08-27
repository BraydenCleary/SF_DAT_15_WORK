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

crime_test = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_test_full.csv')
dummies = pd.get_dummies(crime_test, columns=['DayOfWeek', 'PdDistrict'])
for column_name, column in dummies.transpose().iterrows():
  crime_test[column_name] = column
crime_test['is_weekend'] = crime_test.apply(lambda x: 1 if x['DayOfWeek'] in WEEKEND_DAYS else 0, axis=1)
crime_test['time_of_day_bucket'] = crime_test.apply(lambda x: determine_time_of_day_bucket(x['Dates']), axis=1)
crime_test['day_of_month'] = crime_test.apply(lambda x: datetime.strptime(x['Dates'], '%Y-%m-%d %H:%M:%S').day, axis=1)
crime_test['month_of_year'] = crime_test.apply(lambda x: datetime.strptime(x['Dates'], '%Y-%m-%d %H:%M:%S').month, axis=1)
crime_test['year'] = crime_test.apply(lambda x: datetime.strptime(x['Dates'], '%Y-%m-%d %H:%M:%S').year, axis=1)
