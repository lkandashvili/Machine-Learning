import pandas as pd
from matplotlib import pyplot as plt
from pandas import value_counts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as pt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


data = pd.read_csv("Quiz_2.csv")
y = data["target"]
x = data.drop('target', axis='columns')

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=5)

model = GaussianNB(priors=[0,1])
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.score(X_train,y_train))

print(y.value_counts(normalize=True))