import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt



data = pd.read_excel(r'C:\Users\hp\Desktop\futuremlproject\p1\Project Dataset & Instructions.xlsx',sheet_name="Data")
# print(data.describe)

X = data.drop(columns=["Y", "SampleNo"])
Y = data["Y"]
# print(X)
# print(Y)
x = X.iloc[:100]
y = Y.iloc[:100]
# print(x)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# # Create a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"R2 Score: {r2}")
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
# print(x.shape, X_train.shape, X_test.shape)
model = XGBRegressor()
model.fit(X_train, Y_train)
training_data_prediction = model.predict(X_train)
# print(training_data_prediction)
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()
test_data_prediction = model.predict(X_test)
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)