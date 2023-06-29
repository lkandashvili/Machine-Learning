from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=15, n_informative=3, n_redundant=12, n_classes=3,
                           weights=[0.8, 0.1, 0.1], random_state=1)
print(Counter(y))

model = DecisionTreeClassifier()
parameter = {"max_depth": [4, 5, 6, 7, 8, 9, 10]}
hybrid = GridSearchCV(model, parameter, scoring='accuracy', cv=10, n_jobs=-1, verbose=4)
hybrid.fit(X, y)
print(hybrid.best_score_, hybrid.best_params_)

# scores = cross_val_score(model, X, y, scoring='accuracy', cv=10, n_jobs=-1)
# print(scores.mean())