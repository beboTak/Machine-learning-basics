import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = {
    'account_length': [128, 107, 137, 84, 75, 118, 121, 147, 117, 141],
    'international_plan': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    'voice_mail_plan': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'number_vmail_messages': [25, 26, 0, 0, 0, 0, 0, 0, 0, 0],
    'total_day_minutes': [265.1, 161.6, 243.4, 299.4, 166.7, 223.4, 218.2, 157.0, 184.5, 258.6],
    'total_day_calls': [110, 123, 114, 71, 113, 98, 88, 79, 97, 84],
    'total_eve_minutes': [197.4, 195.5, 121.2, 61.9, 148.3, 220.6, 348.5, 103.1, 252.4, 196.9],
    'total_eve_calls': [99, 103, 110, 88, 122, 101, 108, 94, 115, 105],
    'total_night_minutes': [244.7, 254.4, 162.6, 196.9, 186.9, 203.9, 212.6, 188.8, 143.7, 199.8],
    'total_night_calls': [91, 103, 104, 89, 121, 118, 118, 123, 98, 96],
    'total_intl_minutes': [10.0, 13.7, 12.2, 6.6, 10.1, 6.3, 7.5, 9.9, 8.2, 11.1],
    'total_intl_calls': [3, 3, 5, 7, 3, 6, 7, 9, 4, 6],
    'customer_service_calls': [1, 1, 0, 2, 3, 0, 3, 0, 1, 2],
    'churn': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1]
}

churn_df = pd.DataFrame(data)

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train,y_train)

print(knn.score(X_test, y_test))