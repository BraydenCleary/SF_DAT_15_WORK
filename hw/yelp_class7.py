# 1. Read `yelp.csv` into a DataFrame.
import pandas as pd
yelp_data = pd.read_csv("https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/hw/optional/yelp.csv")
yelp_data.head()

# 2. Explore the relationship between each of the vote types (cool/useful/funny)
# and the number of stars.
pd.scatter_matrix(yelp_data, figsize=(12, 10))

# 3. Define cool/useful/funny as the features, and stars as the response.
X = yelp_data[['cool', 'useful', 'funny']]
y = yelp_data['stars']

# 4. Fit a linear regression model and interpret the coefficients. Do the 
# coefficients make intuitive sense to you? Explore the Yelp website to see if
# you detect similar trends.

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from collections import Counter
import numpy as np

linreg = LinearRegression()
linreg.fit(X, y)
linreg.intercept_ # 3.84
linreg.coef_ # cool: 0.27435947, useful: -0.14745239, funny: -0.13567449

def compute_rmse(X, y):
    feature_train, feature_test, response_train, response_test = train_test_split(X, y, random_state=2)
    linreg.fit(feature_train, response_train)
    y_pred = linreg.predict(feature_test)
    return np.sqrt(mean_squared_error(response_test, y_pred))

# cool, useful, funny
    compute_rmse(yelp_data[['cool', 'useful', 'funny']], y)
    
# cool, useful
    compute_rmse(yelp_data[['cool', 'useful']], y)
    
# cool, funny
    compute_rmse(yelp_data[['cool', 'funny']], y)
    
# useful, funny
    compute_rmse(yelp_data[['useful', 'funny']], y)
    
# Bonus: Think of some new features you could create from the existing data that might be predictive of the response. (This is called "feature engineering".) Figure out how to create those features in Pandas, add them to your model, and see if the RMSE improves.
negative_words = pd.read_csv("https://gist.githubusercontent.com/BraydenCleary/0456409917297af7fbcd/raw/fadc4736c05a2a716f2377ab99447f8532093191/negative_words.csv", names=['negative_word'])['negative_word'].tolist()

def negative_word_count(text):
    negative_count = 0
    text_split_by_spaces = text.split(' ')
    for word in text_split_by_spaces:
        if word in negative_words:
            negative_count += 1
    return negative_count

yelp_data['negative_word_count'] = yelp_data['text'].apply(lambda text: negative_word_count(text))

X = yelp_data[['negative_word_count']]
y = yelp_data['stars']
linreg = LinearRegression()
linreg.fit(X, y)
linreg.intercept_
linreg.coef_

compute_rmse(X, y) # fail

# Bonus: Compare your best RMSE on testing set with the RMSE for the "null model", which is the model that ignores all features and simply predicts the mean rating in the training set for all observations in the testing set.

# Bonus: Instead of treating this as a regression problem, treat it as a classification problem and see what testing accuracy you can achieve with KNN.

# Bonus: Figure out how to use linear regression for classification, and compare its classification accuracy to KNN.
