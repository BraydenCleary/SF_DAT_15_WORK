import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.grid_search import GridSearchCV

glass_data = pd.read_csv('https://gist.githubusercontent.com/BraydenCleary/bf9f0f0f8f5c978c1b32/raw/e864fe6a4b9ac35fdd324696f4470b7588a30403/glass_data.csv')

X, y = glass_data.drop('type', axis = 1), glass_data['type'] 

knn = KNeighborsClassifier()
k_range = range(1, 40)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=50, scoring='accuracy')
grid.fit(X, y)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]

plt.figure()
plt.plot(k_range, grid_mean_scores)

grid.best_score_     
grid.best_params_
grid.best_estimator_ 

#scratch work...messing around:
  #from sklearn.cross_validation import train_test_split

  #features_train, features_test, response_train, response_test = train_test_split(X, y)

  #knn = KNeighborsClassifier(n_neighbors = 4)
  #knn.fit(features_train, response_train)
  #knn.score(features_test, response_test)

  #from sklearn.cross_validation import cross_val_score
  #knn = KNeighborsClassifier(n_neighbors=3)
  #scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
  #print(np.mean(scores))