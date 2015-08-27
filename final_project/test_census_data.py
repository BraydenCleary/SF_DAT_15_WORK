import csv
import pandas as pd
import numpy as np
import requests
import yaml

ASK_GEO_CREDENTIALS = yaml.load(open('credentials.yml', 'r'))['ask_geo']

base_url = 'https://api.askgeo.com/v1/' + ASK_GEO_CREDENTIALS['user_id'] + '/' + ASK_GEO_CREDENTIALS['api_key'] + '/query.json?databases=UsZcta2010&points='

base_url = 'https://api.askgeo.com/v1/1436/542b264a884c83c3f2099bd1bb8a948f360999fa42cf1106c2956b275311bd54/query.json?databases=UsZcta2010&points='
crime_test = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_test.csv')

for index, row in crime_test.iterrows():
  print(index)
  if index < 704111:
    continue
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

test_extra_data = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/zip_test_real.csv')

import re

yo = []
hey = 0
columns = []
with open('zips_test.csv', 'r') as f:
    reader = csv.reader(f)
    for ss in reader:
      if re.search(r'error', ss[0]):
        print('error')
        hey += 2
        continue
      else:
        hey += 1
        print(hey)
        if hey % 2 == 0:
          columns = map(lambda x: x.strip(), ss[1:])
        else:
          row = crime_test.iloc[[int(ss[0])]]
          for idx, column in enumerate(columns[1:]):
            row[column] = ss[1:][idx].strip()

import csv
import urllib2

url = 'https://s3.amazonaws.com/braydencleary-data/final_project/zip_test_real.csv'
response = urllib2.urlopen(url)
cr = csv.reader(response)

hey = 0
for ss in cr:
  hey += 1
  print(hey)
  if len(ss) != 142:
    continue
  if 'CensusAgeMedianAge' in map(lambda x: x.strip(), ss):
    continue
  fd = open('zips_test_with_yo.csv','a')
  print(ss)
  fd.write(', '.join(ss) + '\n')
  fd.close()


hey = 0
for ss in cr:
  hey += 1
  print(hey)
  if len(ss) == 142 or len(ss) == 2:
    continue
  if 'CensusAgeMedianAge' in map(lambda x: x.strip(), ss):
    continue
  fd = open('zips_test_with_139.csv','a')
  print(ss)
  fd.write(', '.join(ss) + '\n')
  fd.close()



for ss in cr:
  if len(ss) == 139:
    print(ss)
    break

