import numpy as np

crime_test = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/crime_test.csv')
first      = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/test_with_142.csv')
first.rename(columns=lambda x: x.strip())
second     = pd.read_csv('https://s3.amazonaws.com/braydencleary-data/final_project/test_with_139.csv')
second.rename(columns=lambda x: x.strip())
first.set_index('test_id', inplace=True)
second.set_index('test_id', inplace=True)
crime_test = pd.merge(crime_test, first, how='left', left_index=True, right_index=True)

k = 0
for row in second.iterrows():
  k += 1
  print(k)
  idx = row[0]
  for index, column in enumerate(map(lambda x: x.strip(), np.array(second.columns[1:]))):
    crime_test.iloc[[idx]][column] = row[1][index]








