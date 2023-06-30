import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.pyplot import scatter


data = pd.read_csv("heart.csv")

y = data["target"].values
X = data.drop("target", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
PCA = PCA(n_components=6)
Xnew = PCA.fit_transform(X)

modelForTest = RandomForestClassifier()
modelForTrain = RandomForestClassifier()
modelForXnew = RandomForestClassifier(criterion="entropy")

modelForTest.fit(X_test, y_test)
modelForTrain.fit(X_train, y_train)
modelForXnew.fit(Xnew, y)

print(modelForTest.score(X_test, y_test))
print(modelForTrain.score(X_train, y_train))
print(modelForXnew.score(Xnew, y))


data2 = pd.read_csv("Cars.csv")
# fueltype = pd.get_dummies(data2["fueltype"])
# aspiration = pd.get_dummies(data2["aspiration"])
# data2 = pd.concat([fueltype, aspiration], axis=1)
y2 = data2["price"].values
X2 = data2.drop("price", axis=1).values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=5)

model2ForTest = LinearRegression()
model2ForTrain = LinearRegression()
model2ForTest.fit(X2_test, y2_test)
model2ForTrain.fit(X2_train, y2_train)

print(model2ForTest.score(X2_test, y2_test))
print(model2ForTrain.score(X2_train, y2_train))
print(data2)


data3 = pd.read_csv("Cluster.csv")

X3 = data3["Feature1"].values
y3 = data3["Feature2"].values
scatter = scatter(X3, y3)
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(scatter)







