import pandas as pd
# pd.set_option('max_colwidth', 50)
# set this if you need to

import sys,os 
path = os.path.realpath('../hw/data/police-killings.csv')
killings = pd.read_csv(path)
killings.head()

# 1. Make the following changed to column names:
# lawenforcementagency -> agency
# raceethnicity        -> race

  killings.rename(columns={'lawenforcementagency':'agency', 'raceethnicity':'race'}, inplace=True)

# 2. Show the count of missing values in each column

  killings.isnull().sum()
  # Four missing street addresses

# 3. replace each null value in the dataframe with the string "Unknown"

  killings.fillna('Unknown', inplace=True)
  killings.isnull().sum()

# 4. How many killings were there so far in 2015?

  killings[killings['year'] == 2015].name.count()
  # 467

# 5. Of all killings, how many were male and how many female?

  killings.groupby('gender').count().state
  # 22 female, 445 male

# 6. How many killings were of unarmed people?

  # killings.armed.value_counts()
  # I'm assuming "No" value in armed column equates to being unarmed
  killings[killings['armed'] == 'No'].count().armed
  # 102 unarmed people killed by cops

# 7. What percentage of all killings were unarmed?
  (float(killings[killings['armed'] == 'No'].count().armed) / killings[killings['armed'] != 'No'].count().armed) * 100
  # 27.95%

# 8. What are the 5 states with the most killings?

  killings.groupby('state').size().order(ascending=False).head(5)
  # CA, TX, FL, AZ, OK

# 9. Show a value counts of deaths for each race
  killings.groupby('race').size().order(ascending=False)

# 10. Display a histogram of ages of all killings

  killings['age'].hist()

# 11. Show 6 histograms of ages by race

  killings['age'].hist(by=killings['race'], sharex=True, sharey=True)

# 12. What is the average age of death by race?

  killings.groupby('race').age.mean()

# 13. Show a bar chart with counts of deaths every month

  killings.month.value_counts().plot(kind='bar')


###################
### Less Morbid ###
###################


import pandas as pd
import sys,os 
path = os.path.realpath('../hw/data/college-majors.csv')
majors = pd.read_csv(path)
majors.head()

# 1. Delete the columns (employed_full_time_year_round, major_code)
  majors.drop(['Employed_full_time_year_round', 'Major_code'], axis=1, inplace=True)

# 2. Show the cout of missing values in each column

  majors.isnull().sum()

# 3. What are the top 10 highest paying majors?
  # Using median column as a proxy for amount a major makes
  majors.sort_index(by='Median', ascending=False).head(10)['Major']

# 4. Plot the data from the last question in a bar chart, include proper title, and labels!
  top_10_majors_by_median = majors.sort_index(by='Median', ascending=False).head(10)[['Major', 'Median']]
  graph = top_10_majors_by_median.plot(kind='bar', x='Major', y='Median', title='Top Paying Majors')
  graph.set_xlabel('Major',fontsize=12)
  graph.set_ylabel('Median',fontsize=12)
  
# 5. What is the average median salary for each major category?
  
  sorted_major_category_by_mean_of_median = majors.groupby('Major_category').Median.mean().order(ascending=False)

# 6. Show only the top 5 paying major categories
  sorted_major_category_by_mean_of_median.head(5)
  
# 7. Plot a histogram of the distribution of median salaries
  
  hist = majors['Median'].hist()
  hist.set_xlabel('Median')
  hist.set_ylabel('Count')

# 8. Create a bar chart showing average median salaries for each major_category
  majors.groupby('Major_category')['Median'].mean().order().plot(kind='bar')
  
# 9. What are the top 10 most UNemployed majors?
# What are the unemployment rates?
  majors.sort_index(by='Unemployment_rate', ascending=False)[['Major', 'Unemployment_rate']].head(10)

# 10. What are the top 10 most UNemployed majors CATEGORIES? Use the mean for each category
# What are the unemployment rates?
  majors.groupby('Major_category')['Unemployment_rate'].mean().order(ascending=False).head(10)

# 11. the total and employed column refer to the people that were surveyed.
# Create a new column showing the emlpoyment rate of the people surveyed for each major
# call it "sample_employment_rate"
# Example the first row has total: 128148 and employed: 90245. it's
# sample_employment_rate should be 90245.0 / 128148.0 = .7042
  majors['sample_employment_rate'] = majors['Employed'] / majors['Total']

# 12. Create a "sample_unemployment_rate" colun
# this column should be 1 - "sample_employment_rate"
  majors['sample_unemployment_rate'] = 1 - majors['sample_employment_rate']
