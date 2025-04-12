from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd

churn_df = pd.read_csv("telecom_churn_clean.csv")

 
X = churn_df[["total_day_charge", "total_eve_charge"]].values 
y = churn_df["churn"].values  

logreg = LogisticRegression() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
random_state=42)  
logreg.fit(X_train, y_train) 
y_pred = logreg.predict(X_test) 

y_pred_probs = logreg.predict_proba(X_test)[:, 1] 
print(y_pred_probs[0]) 

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs) 
plt.plot([0, 1], [0, 1], 'k--') 
plt.plot(fpr, tpr) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('Logistic Regression ROC Curve') 
plt.show() 