import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_classes=2,
    random_state=42,
    weights=[0.8, 0.2]  
)


churn_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
churn_df['churn'] = y


X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


print(f"Initial KNN accuracy (k=5): {knn.score(X_test, y_test):.2f}")

neighbors = np.arange(1, 21)  
train_accuracies = {}
test_accuracies = {}


for neighbor in neighbors:
   
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    
    knn.fit(X_train, y_train)
    
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("\nNeighbors:", neighbors)
print("Training accuracies:", train_accuracies)
print("Testing accuracies:", test_accuracies)

plt.figure(figsize=(10, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, list(train_accuracies.values()), label="Training Accuracy", marker='o')
plt.plot(neighbors, list(test_accuracies.values()), label="Testing Accuracy", marker='o')
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.grid(True)
plt.show()