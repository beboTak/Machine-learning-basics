from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


data = pd.read_csv("telecom_churn_clean.csv")

X = data.drop("churn" ,axis=1)
y = data["churn"].values

print(X.shape)
print(y.shape)

X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=42,stratify=y)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

knn = KNeighborsClassifier(n_neighbors=16)

knn.fit(X_train,y_train)

print(knn.score(X_test, y_test)) 