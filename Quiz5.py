"""
1.	მოცემულია ფაილი weatherAUS.csv , რომელშიც სვეტი RainTomorrow  არის  დამოკიდებული სვეტი(target ) , ხოლო ყველა სხვა ცვლადი არის დამოუკიდებელი : უპასუხეთ ქვევით მოცემულ კითხვებს (10 ქულა)
•	სვეტიდან  Date ამოიღეთ  ინფორმაცია თვის შესახებ  და შექმენით სვეტი Month. Date სვეტი წაშალეთ ამის შემდეგ (2 ქულა)

•	ყველა გამოტოვებული ელემენტი შეავსეთ საშუალოთი(რიცხვითი  სვეტის) და  მოდათი ტექსტური სვეტი(1 ქულა)

•	გადაიყვანეთ ყველა ტექსტური სვეტი რიცხვითში (1 ქულა)

•	VarianceThreshold  ის მეთოდის მეშვეობით შეარჩიეთ  სვეტები -რომლებიც დააკმაყოფილებენ  თქვენს მიერ მითითებული threshold ის  ზღვარს -აღწერეთ თუ  რას განსაზღვრავს  threshold ის ბარიერი და  რამდენად მაღალ შედეგს მოგცემთ   ჩატარებული ექსპერიმენტ(ალგორითმი კლასიფიკაციის ამოარჩიეთ თქვენი სურვილით, ასევე  სატრენინგო და სატესტო მონაცემების ზომაც აიღეთ თქვენით) (6 ქულა)g
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv("weatherAUS.csv", sep=",")

y = data["RainTomorrow"]
X = data.drop("RainTomorrow", axis=1)

data["Month"] = data["Date"].map(lambda x: int(x.split('/')[0]))
data.drop("Date", axis=1, inplace=True)

myLabel = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object' or data[column].dtype == bool:
        data[column] = myLabel.fit_transform(data[column])

selector = VarianceThreshold(threshold=10)
selector.fit(data)
print(selector.get_support())

data = data.iloc[:, selector.get_support()]
print(data)