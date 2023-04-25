import html5lib as html5lib
import lxml as lxml
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv",header=None)
# print(dataset.head())
# print(dataset.corr())

y = dataset.iloc[:, 9]
X = dataset.iloc[:, 0:9]
model = KNeighborsClassifier()

model.fit(X,y)
print(model.score(X,y))

PCA=PCA(n_components=5)
Xnew = PCA.fit_transform(X)
print(X.shape, Xnew.shape)

model.fit(Xnew,y)
print(model.score(Xnew,y))

print(np.sum(PCA.explained_variance_ratio_))

