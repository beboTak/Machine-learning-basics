from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge
import numpy as np
import pandas as pd

diabetes_df = pd.read_csv("diabetes_clean.csv") 
print(diabetes_df.head()) 


X = diabetes_df.drop("glucose", axis=1).values 
y = diabetes_df["glucose"].values  
  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
kf = KFold(n_splits=6, shuffle=True, random_state=42) 
reg = LinearRegression() 
cv_results = cross_val_score(reg, X, y, cv=kf) 

print(cv_results) 

print(np.mean(cv_results), np.std(cv_results)) 

print(np.quantile(cv_results, [0.025, 0.975])) 


scores = [] 
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]: 
  ridge = Ridge(alpha=alpha) 
  ridge.fit(X_train, y_train) 
  y_pred = ridge.predict(X_test) 
  scores.append(ridge.score(X_test, y_test)) 
print(scores) 



scores = [] 
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]: 
 lasso = Lasso(alpha=alpha) 
 lasso.fit(X_train, y_train) 
 lasso_pred = lasso.predict(X_test) 
 scores.append(lasso.score(X_test, y_test)) 
print(scores) 