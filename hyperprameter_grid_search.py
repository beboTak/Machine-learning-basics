from sklearn.model_selection import GridSearchCV,KFold,train_test_split

from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
kf = KFold(n_splits=5, shuffle=True, random_state=42)  
param_grid = {'alpha': np.arange(0.0001, 1, 10), 'solver': ['sag', 'tsqr']}  
diabetes_df = pd.read_csv("diabetes_clean.csv")
X = diabetes_df.drop("glucose",axis=1).values
y = diabetes_df["glucose"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
ridge = Ridge()  
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)  
ridge_cv.fit(X_train, y_train)  
print(ridge_cv.best_params_, ridge_cv.best_score_)  