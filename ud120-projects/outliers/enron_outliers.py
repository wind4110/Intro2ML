#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("ud120-projects/final_project/final_project_dataset.pkl", "rb") )
data_dict.pop("TOTAL", 0) # remove outlier
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below



from sklearn.linear_model import LinearRegression
import numpy as np
reg = LinearRegression()
salary = data[:, 0].reshape(-1, 1)
bonus = data[:, 1].reshape(-1, 1)
reg.fit(salary, bonus)
print("Slope (coefficient):", reg.coef_)
print("Intercept:", reg.intercept_)
print("R^2 score:", reg.score(salary, bonus))
# Identify the outlier
max_salary = np.max(salary)
outlier = [name for name, features in data_dict.items() if features["salary"] == max_salary]
print("Outlier:", outlier)

for name, features in data_dict.items():
    if features["salary"] == "NaN" or features["bonus"] == "NaN":
        continue
    if features["salary"] > 1000000 and features["bonus"] > 5000000:
        print("Outlier details:", name, np.abs(features['bonus']-reg.predict([[features["salary"]]])[0][0]))



plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("salary")
plt.ylabel("bonus")
plt.plot(salary, reg.predict(salary), color="blue")
plt.show()
