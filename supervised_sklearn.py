import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = {
    'account_length': [128, 107, 137, 84, 75, 118, 121, 147, 117, 141],
    'customer_service_calls': [1, 1, 0, 2, 3, 0, 3, 1, 0, 2],
    'churn': [0, 0, 0, 0, 1, 0, 1, 0, 0, 1]  
}


churn_df = pd.DataFrame(data)
X = churn_df[["account_length", "customer_service_calls"]].values
y = churn_df["churn"].values

knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X, y)
X_new = np.array([
    [120, 2],   
    [115, 3],    
    [150, 1]    
])


predictions = knn.predict(X_new)

print("Training Data:")
print(churn_df)
print("\nPredictions for New Data:")
for i, pred in enumerate(predictions):
    print(f"Customer {i+1}: {'Will Churn' if pred == 1 else 'Will Not Churn'}")
    