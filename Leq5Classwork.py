import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as pt

salary = pd.read_csv("https://raw.githubusercontent.com/erikgregorywebb/datasets/master/nba-salaries.csv")
myLabel = LabelEncoder()
salary['position'] = myLabel.fit_transform(salary['position'])
Teams = pd.get_dummies(salary['team'])
salary = pd.concat([salary,Teams],axis=1)
salary.drop(["team","rank","name"],axis=1, inplace=True)

print(salary.head())

y = salary['salary'].values
x = salary.drop("salary", axis=1).values
model = LinearRegression()
# x = x.reshape(-1, 1)
model.fit(x, y)
print(model.score(x, y))
# y_predicted = model.predict(x)
# plt.scatter(x, y)
# plt.plot(x,y)
# plt.show()


