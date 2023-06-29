"""
1. მოცემულია ფაილი Quiz_2.csv, რომელშიც დამოკიდებული ცვლადი არის target ,
ხოლო ყველა სხვა ცვლადი არის დამოუკიდებელი ცვლადი, უპასუხეთ ქვევით
მოცემულ კითხვებს(10 ქულა)
 დაყავით მონაცემები სატრენინგოდ და სატესტოდ და გამოიყენეთ
GaussianNB ის ალგორითმი და გამოთვალეთ როგორც სატრენინგო , ასევე
სატესტო Score (4 ქულა)

 ფუნქცია value_counts() ის მეშვეობით გაიგეთ თავდაპირველ target სვეტში
როგორ არის კლასები განაწილებული პროპროციულად(normalize=True
დამხმარე ფუნქციის მეშვეობით value_counts() ში) და GaussianNB
ალგორითმში priors პარამეტრის მეშვეობით გაუწერეთ სასურველი ალბათობები და შეამოწმეთ როგორ შეიცვლება შედეგები. აღწერეთ
მიღებული შედეგი (6 ქულა)
"""

from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Quiz_2.csv")

y = data["target"].values
X = data.drop("target", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
model = GaussianNB(priors=[0.5, 0.5])
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

print(data["target"].value_counts(normalize=True))