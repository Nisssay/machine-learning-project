import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_excel(r'C:\Users\hp\Desktop\futuremlproject\p1\Project Dataset & Instructions.xlsx', sheet_name="Data")

# Split the data into features (X) and target variable (Y)
X = data.drop(columns=["Y", "SampleNo"])
Y = data["Y"]
x = X.iloc[:100]
y = Y.iloc[:100]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train and evaluate Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
print("Linear Regression:")
print(f"Mean Squared Error: {lr_mse}")
print(f"R2 Score: {lr_r2}")

# Train and evaluate Decision Tree Regression
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)
print("Decision Tree Regression:")
print(f"Mean Squared Error: {dt_mse}")
print(f"R2 Score: {dt_r2}")

# Train and evaluate Random Forest Regression
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print("Random Forest Regression:")
print(f"Mean Squared Error: {rf_mse}")
print(f"R2 Score: {rf_r2}")

# Train and evaluate Support Vector Regression
svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_y_pred = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)
print("Support Vector Regression:")
print(f"Mean Squared Error: {svr_mse}")
print(f"R2 Score: {svr_r2}")
