##### Part 1 #####

# 1. read in the yelp dataset

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

yelp_data = pd.read_csv("https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/hw/optional/yelp.csv")
yelp_data.head()

# 2. Perform a linear regression using 
# "stars" as your response and 
# "cool", "useful", and "funny" as predictors

linreg = LinearRegression()
X = yelp_data[['cool', 'useful', 'funny']]
y = yelp_data['stars']

features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=3)
linreg.fit(features_train, response_train)

# from sklearn.cross_validation import cross_val_score
# scores = cross_val_score(linreg, features_test, response_test, cv=30)

# 3. Show your MAE, R_Squared and RMSE

y_pred = linreg.predict(features_test)
mae    = metrics.mean_absolute_error(response_test, y_pred)
r2     = metrics.r2_score(response_test, y_pred)
rmse   = np.sqrt(metrics.mean_squared_error(response_test, y_pred))

# 4. Use statsmodels to show your pvalues
# for each of the three predictors
# Using a .05 confidence level, 
# Should we eliminate any of the three?

lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp_data).fit()
lm.pvalues

# no, we shouldn't eliminate any of the three since all p values are less than .05

# 5. Create a new column called "good_rating"
# this could column should be True iff stars is 4 or 5
# and False iff stars is below 4

def good_or_not(stars):
    if stars >= 4:
        return True
    else:
        return False

yelp_data['good_rating'] = yelp_data.stars.apply(good_or_not)

# 6. Perform a Logistic Regression using 
# "good_rating" as your response and the same
# three predictors

logreg = LogisticRegression()
X = yelp_data[['cool', 'useful', 'funny']]
y = yelp_data['good_rating']
features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=3)
logreg.fit(features_train, response_train)
y_pred = logreg.predict(features_test)

# 7. Show your Accuracy, Sensitivity, Specificity
# and Confusion Matrix

def logreg_metrics(cmat):
    # assumes 2x2 confusion matrix
    # returns sensitivity, specificity, and accuracy
    return [cmat[1][1] / float(cmat[0][1] + cmat[1][1]), cmat[0][0] / float(cmat[0][0] + cmat[1][0]), (cmat[0][0] + cmat[1][1]) / float((cmat.sum()))]

cmat = confusion_matrix(response_test, y_pred)
logreg_metrics(cmat)

# 8. Perform one NEW operation of your 
# choosing to try to boost your metrics!

# Add feature that counts how many times "love" appears in text review
def love_count(text):
    love_count = 0
    text_split_by_spaces = text.split(' ')
    for word in text_split_by_spaces:
        if word == 'love':
            love_count += 1
    return love_count
    
yelp_data['love_count'] = yelp_data['text'].apply(lambda text: love_count(text))

logreg = LogisticRegression()
X = yelp_data[['cool', 'useful', 'funny', 'love_count']]
y = yelp_data['good_rating']
features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=3)
logreg.fit(features_train, response_train)
y_pred = logreg.predict(features_test)
cmat = confusion_matrix(response_test, y_pred)

logreg_metrics(cmat)

##### Part 2 ######

# 1. Read in the titanic data set.

titanic_data = pd.read_csv('https://gist.githubusercontent.com/BraydenCleary/1e2f4ab267b3d7a94e98/raw/3d2e1547ec3f2f4ea07c4c8077e00c184ca559df/titanic.csv')

# 4. Create a new column called "wife" that is True
# if the name of the person contains Mrs.
# AND their SibSp is at least 1

def is_wife(name, sibsp):
    if 'Mrs.' in name and sibsp >= 1:
        return True
    else:
        return False

titanic_data['wife'] = titanic_data.apply(lambda x: is_wife(x['Name'], x['SibSp']), axis=1)

# 5. What is the average age of a male and
# the average age of a female on board?

avg_male_age = titanic_data[titanic_data['Sex'] == 'male']['Age'].mean()
avg_female_age = titanic_data[titanic_data['Sex'] == 'female']['Age'].mean()

# 5. Fill in missing MALE age values with the
# average age of the remaining MALE ages

# see below

# 6. Fill in missing FEMALE age values with the
# average age of the remaining FEMALE ages

def set_null_age_as_mean(row):
    if row['Age'] > 0:
        pass
    else:
        if row['Sex'] == 'male':
            row['Age'] = avg_male_age
        elif row['Sex'] == 'female':
            row['Age'] = avg_female_age
    return row

    
titanic_data = titanic_data.apply(set_null_age_as_mean, axis=1)

# 7. Perform a Logistic Regression using
# Survived as your response and age, wife
# as predictors

logreg = LogisticRegression()
X = titanic_data[['Age', 'wife']]
y = titanic_data['Survived']
features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=3)
logreg.fit(features_train, response_train)
y_pred = logreg.predict(features_test)

# 8. Show Accuracy, Sensitivity, Specificity and 
# Confusion matrix

cmat = confusion_matrix(response_test, y_pred)
logreg_metrics(cmat)


# 9. now use ANY of your variables as predictors
# Still using survived as a response to boost metrics!

def is_child(name, sibsp, age):
    if 'Mr.' in name or 'Mrs.' in name:
        return False
    elif sibsp >= 1 and age < 13:
        return True
    else:
        return False

titanic_data['is_child'] = titanic_data.apply(lambda x: is_child(x['Name'], x['SibSp'], x['Age']), axis=1)
    
logreg = LogisticRegression()
X = titanic_data[['Age', 'wife', 'Parch', 'SibSp', 'is_child']]
y = titanic_data['Survived']
features_train, features_test, response_train, response_test = train_test_split(X, y, random_state=3)
logreg.fit(features_train, response_train)
y_pred = logreg.predict(features_test)

# 10. Show Accuracy, Sensitivity, Specificity

cmat = confusion_matrix(response_test, y_pred)
logreg_metrics(cmat)


# REMEMBER TO USE
# TRAIN TEST SPLIT AND CROSS VALIDATION
# FOR ALL METRIC EVALUATION!!!!

