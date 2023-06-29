"""
1.	მიჰყევით ინსტრუქციებს  და გაუშვით თვითოეული კოდი მიყოლებით, ასევე უპასუხეთ ქვევით მოცემულ კითხვებს (10 ქულა)
•	მოცემულია ქვევით სიის ტიპის მასივი, რომელიც შეიცავს ტექსტებს :
documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]
მოცემული  თვითოეული ტექსტი აღწერს გარკვეული შინაარსის მომენტს.(kitty არის კნუტი).განსაღვრეთ თუ  მოცემულ დოკუმენტის ტექსტებში რამდენ განსხვავებულ მომენტზე არის საუბარი  და KMeans  ის კლასტერებში გაუწერეთ ის რაოდენობა .მანამდე  მიჰყევით ქვევით მოცემულ  ინსტრუქციას :
•	გამოიძახეთ შემდეგი ბიბლიოთეკები :
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
•	გადაიყვანეთ ტექსტები  რიცხვებში შემდეგი ფუნქციის გამოყენებით(ითვლის  სიტყვების სიხშირეების შესაბამის  წონებს)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
•	კლასტერინგისთვის  გამოიყენეთ შემდეგი ბრძანება :
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
სადაც true_k არის თქვენს მიერ შერჩეული კლასტერების რაოდენობა
•	მოახდინეთ პროგნოზირება შემდეგი  თქვენს მიერ შერჩეული ტექსტის შემდეგი ბრანების გამოყენებით :
Y = vectorizer.transform([write your text in  here])
prediction = model.predict(Y)
print(prediction)
კვადრატულ ფრჩხილებში უნდა ჩაწეროთ ტექსტი ბრწყალებში (მოიფიქრეთ ტექსტი ისე რომ  ახლოს იყოს დატრენინგებულ ტექსტებთან)


"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

Y = vectorizer.transform(["My mom loves cats."])
prediction = model.predict(Y)
print(prediction)