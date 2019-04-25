from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model.logistic import LogisticRegression

categories = ['rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']

x1 = fetch_20newsgroups(subset='train',categories=categories, remove=('headers', 'footers', 'quotes'))
x2 = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(x1.data)
X_test = vectorizer.transform(x2.data)

classifier = LogisticRegression()
classifier.fit(X_train, x1.target)
predictions = classifier.predict(X_test)
print (classification_report(x2.target, predictions))

classifier = Perceptron(n_iter=100, eta0=0.1)
classifier.fit(X_train, x1.target )
predictions = classifier.predict(X_test)
print (classification_report(x2.target, predictions))



