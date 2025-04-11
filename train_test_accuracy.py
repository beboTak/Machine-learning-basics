from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

churn_df = pd.read_csv("telecom_churn_clean.csv")

print(churn_df.head)
print(churn_df.info)

X = churn_df.drop("churn",axis=1).values
y = churn_df["churn"].values
print(X.shape)
print(y.shape)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
train_accuracies = {} 
test_accuracies = {} 
neighbors = np.arange(1, 26) 
for neighbor in neighbors: 
  knn = KNeighborsClassifier(n_neighbors=neighbor) 
  knn.fit(X_train, y_train) 
  train_accuracies[neighbor] = knn.score(X_train, y_train) 
  test_accuracies[neighbor] = knn.score(X_test, y_test) 


plt.figure(figsize=(8, 6)) 
plt.title("KNN: Varying Number of Neighbors") 
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy") 
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy") 
plt.legend() 
plt.xlabel("Number of Neighbors") 
plt.ylabel("Accuracy") 
plt.show() 