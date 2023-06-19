import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv(r'C:\Users\hp\Desktop\mltraining\Gold Price Prediction\gld_price_data.csv')
# print(gold_data.head())
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
# training the model
regressor.fit(X_train,Y_train)
# prediction on Test Data
test_data_prediction = regressor.predict(X_test)
# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)