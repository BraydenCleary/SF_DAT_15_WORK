import csv
import pandas as pd
import numpy as np
import requests
import yaml

ASK_GEO_CREDENTIALS = yaml.load(open('credentials.yml', 'r'))['ask_geo']

base_url = 'https://api.askgeo.com/v1/' + ASK_GEO_CREDENTIALS['user_id'] + '/' + ASK_GEO_CREDENTIALS['api_key'] + '/query.json?databases=UsZcta2010&points='

crime_train = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_train.csv')
zips        = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/zips_train.csv')
zips.drop_duplicates(subset='index_of_crime_train', inplace=True)
zips.set_index('index_of_crime_train', inplace=True)
crime_train = pd.merge(crime_train, zips, how='inner', left_index=True, right_index=True)

unique_zips = crime_train[['Y', 'X', ' zip']].groupby(' zip').head(1)

for index, row in unique_zips.iterrows():
  try:
    url = base_url + str(row['Y']) + '%2C' + str(row['X'])
    census_data = ', '.join(map(lambda x: str(x), requests.get(url).json()['data'][0]['UsZcta2010'].values())) + '\n'
  except Exception as e:
    print(e)
    census_data = str(0) + '\n'
    fd = open('additional_data_for_zips_train.csv','a')
    fd.write(census_data)
    fd.close()
  fd = open('additional_data_for_zips_train.csv','a')
  fd.write(census_data)
  fd.close()
