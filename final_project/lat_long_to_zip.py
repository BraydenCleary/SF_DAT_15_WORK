import csv
import pandas as pd
import numpy as np
import requests
import yaml

ASK_GEO_CREDENTIALS = yaml.load(open('credentials.yml', 'r'))['ask_geo']

base_url = 'https://api.askgeo.com/v1/' + ASK_GEO_CREDENTIALS['user_id'] + '/' + ASK_GEO_CREDENTIALS['api_key'] + '/query.json?databases=UsZcta2010&points='

lat_long = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/lat_long.csv')

for index, row in lat_long.iterrows():
  try:
    url = base_url + str(row['Y']) + '%2C' + str(row['X'])
    zip = str(requests.get(url).json()['data'][0]['UsZcta2010']['ZctaCode']) + '\n'
  except Exception as e:
    print(e)
    zip = str(0) + '\n'
    fd = open('zips.csv','a')
    fd.write(str(index) + ', ' + zip)
    fd.close()
  fd = open('zips.csv','a')
  fd.write(str(index) + ', ' + zip)
  fd.close()
