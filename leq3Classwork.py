
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv",header=None)
print(data.head())
target = data.iloc[:,7].values
X = data.iloc[:,0:7].values
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=1)

myModel = LinearSVC(max_iter=20000)
myModel.fit(X_train, y_train)
print(myModel.score(X_test, y_test))