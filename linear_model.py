from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
diabetes_df = pd.read_csv("diabetes_clean.csv") 
print(diabetes_df.head()) 


X = diabetes_df.drop("glucose", axis=1).values 
y = diabetes_df["glucose"].values  

reg = LinearRegression()


reg.fit(X,y)


predictions = reg.predict(X)

print(predictions[:5])

###################################################################################
###################################################################################
###################################################################################
###################################################################################

import matplotlib.pyplot as plt

plt.scatter(X, y, color="blue")

plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

plt.show()


###################################################################################
###################################################################################
###################################################################################
###################################################################################


sales_df = pd.read_csv("advertising_and_sales_clean.csv")
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()

reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))


###################################################################################
###################################################################################
###################################################################################
###################################################################################


from sklearn.metrics import root_mean_squared_error

r_squared = reg.score(X_test, y_test)

rmse = root_mean_squared_error(y_test, y_pred, squared=False)


print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))


###################################################################################
###################################################################################
###################################################################################
###################################################################################


from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()


cv_scores = cross_val_score(reg,X, y, cv=kf)


print(cv_scores)


