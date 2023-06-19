import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv(r'C:\Users\hp\Desktop\mltraining\Calories Burnt Prediction\calories.csv')
# print(calories.shape)
# print(calories.isnull().sum())
exercise_data = pd.read_csv(r'C:\Users\hp\Desktop\mltraining\Calories Burnt Prediction\exercise.csv')
# print(exercise_data.shape)
# print(exercise_data.isnull().sum())
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
# print(calories_data.head())
correlation = calories_data.corr()
# constructing a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
# plt.show()
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
# print(calories_data.head())
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# loading the model
model = XGBRegressor()
# training the model with X_train
model.fit(X_train, Y_train)
test_data_prediction = model.predict(X_test)
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)