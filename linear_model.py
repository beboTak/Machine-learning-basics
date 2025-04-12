
from sklearn.linear_model import LinearRegression
import pandas as pd
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

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")a
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()


###################################################################################
###################################################################################
###################################################################################
###################################################################################


# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train,y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))


###################################################################################
###################################################################################
###################################################################################
###################################################################################

# Import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = root_mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))


###################################################################################
###################################################################################
###################################################################################
###################################################################################

# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

# Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg,X, y, cv=kf)

# Print scores
print(cv_scores)


###################################################################################
###################################################################################
###################################################################################
###################################################################################

# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))