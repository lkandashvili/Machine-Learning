# vowel classification

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Vowel.csv")
data["V1"] = data['V1'].map(int)
data['Class'] = LabelEncoder().fit_transform(data['Class'])
print(data.head())

y = data['Class'].values
X = data.drop("Class", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model = MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), batch_size=30, learning_rate_init=0.01, verbose=1, activation='logistic')
model.fit(X_train, y_train)