import csv
import pandas as pd
import numpy as np
import requests
import yaml

ASK_GEO_CREDENTIALS = yaml.load(open('credentials.yml', 'r'))['ask_geo']

base_url = 'https://api.askgeo.com/v1/' + ASK_GEO_CREDENTIALS['user_id'] + '/' + ASK_GEO_CREDENTIALS['api_key'] + '/query.json?databases=UsZcta2010&points='

crime_test = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_test.csv')

for index, row in crime_test.iterrows():
  try:
    url = base_url + str(row['Y']) + '%2C' + str(row['X'])
    data = requests.get(url).json()['data'][0]['UsZcta2010']
    headers = str(index) + ',' + ', '.join(str(x) for x in data.keys()) + '\n'
    values = str(index) + ',' + ', '.join(str(x) for x in data.values()) + '\n'
    fd = open('zips_test.csv','a')
    fd.write(headers)
    fd.write(values)
    fd.close()
  except Exception as e:
    print(e)
    fd = open('zips_test.csv','a')
    fd.write(str(index) + ' error \n')
    fd.close()
