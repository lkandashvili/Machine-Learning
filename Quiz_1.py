"""
სპეციალური ბიბლიოთეკის sklearn.svm ის გამოყენებით ჩატვირთვეთ ორი
    ალგორითმი : LinearSVC და SVC და უპასუხეთ ქვევით მოცემულ
    კითხვებს(10 ქულა)
     დაყავით მონაცემები სატრენინგო და სატესტო ნაწილებად
    train_test_split ფუნქციის გამოყენებით ბიბლიოთეკიდან
    sklearn.model_selection .სატესტო ზომად იყოს 10% (2 ქულა)

 გამოიყენეთ LinearSVC ის ალგორითმი და გამოთვალეთ
სატრენინგო და სატესტო score და შეადარეთ ერთმანეთს (2 ქულა)

 გამოიყენეთ SVC ის ალგორითმი -რომლსაც შიდა პარამეტრების
სახით გაუწერეთ : kernel როგორც ‘sigmoid’ და gamma როგორც
float რიცხვი. გამოთვალეთ სატრენინგო და სატესტო score და
შეადარეთ წინა ალგორითმს(6 ქულა)
"""


import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv", header=None)
target =data.iloc[:,7].values
X =data.iloc[:,0:7].values

X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.1, random_state=2)
# myModel = LinearSVC(max_iter=90000)
# myModel.fit(X_train, y_train)
# print(myModel.score(X_test, y_test))
# print(myModel.score(X_train,y_train))

modelForTest = SVC(kernel='sigmoid', gamma=0.2)
modelForTrain = SVC(kernel='sigmoid', gamma=0.2)
modelForTest.fit(X_test, y_test)
modelForTrain.fit(X_train, y_train)
print(modelForTest.score(X_test,y_test))
print(modelForTrain.score(X_train, y_train))