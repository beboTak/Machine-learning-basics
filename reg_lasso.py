from sklearn.linear_model import Lasso 
import matplotlib.pyplot as plt
import pandas as pd


diabetes_df = pd.read_csv("diabetes_clean.csv")
X = diabetes_df.drop("glucose", axis=1).values 
y = diabetes_df["glucose"].values 


names = diabetes_df.drop("glucose", axis=1).columns 
lasso = Lasso(alpha=0.1) 
lasso_coef = lasso.fit(X, y).coef_ 
plt.bar(names, lasso_coef) 
plt.xticks(rotation=45) 
plt.show() 