import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_excel(r'C:\Users\hp\Desktop\futuremlproject\p1\Project Dataset & Instructions.xlsx', sheet_name="Data")

# Split the data into features (X) and target variable (Y)
X = data.drop(columns=["Y", "SampleNo"])
Y = data["Y"]
x = X.iloc[:100]
y = Y.iloc[:100]

# Split the data into training and testing sets with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train and evaluate Decision Tree Regression with fixed random state and other parameters
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)  # Adjust max_depth as needed
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)

# print("Decision Tree Regression:")
# print(f"Mean Squared Error: {dt_mse}")
# print(f"R2 Score: {dt_r2}")

a=X.iloc[100:]
b=Y.iloc[100:]
# print(a)
# print(b)


# Predict the target variable (Y) for the new data (a)
y_pred = dt_model.predict(a)
print(y_pred)

# Insert the predicted values into the Excel file
data.loc[100:, 'Y'] = y_pred

# Save the updated data with the predicted values to a new Excel file
data.to_excel(r'C:\Users\hp\Desktop\futuremlproject\p1\Project Dataset & Instructions.xlsx', index=False)