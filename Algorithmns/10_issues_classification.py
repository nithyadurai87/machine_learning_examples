# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
# https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b

import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('./Consumer_Complaints.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
df = df[pd.notnull(df['Issue'])]

fig = plt.figure(figsize=(8,6))
df.groupby('Product').Issue.count().plot.bar(ylim=0)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df['Issue'], df['Product'], random_state = 0)
c = CountVectorizer()
clf = MultinomialNB().fit (TfidfTransformer().fit_transform(c.fit_transform(X_train)), y_train)

print(clf.predict(c.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Issue).toarray()
print (features)
df['category_id'] = df['Product'].factorize()[0]
pro_cat = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
print (pro_cat)
for i, j in sorted(dict(pro_cat.values).items()):
	indices = np.argsort(chi2(features, df.category_id == j)[0])
	print (indices)
	feature_names = np.array(tfidf.get_feature_names())[indices]
	unigrams = [i for i in feature_names if len(i.split(' ')) == 1]
	bigrams = [i for i in feature_names if len(i.split(' ')) == 2]
	print(">",i)
	print("unigrams:",','.join(unigrams[:5]))
	print("bigrams:",','.join(bigrams[:5]))



"""


print (features.shape)


df['category_id'] = df['Product'].factorize()[0]
pro_cat = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
for i, j in sorted(dict(pro_cat.values).items()):
	indices = np.argsort(chi2(features, df.category_id == j)[0])
	feature_names = np.array(tfidf.get_feature_names())[indices]
	unigrams = [i for i in feature_names if len(i.split(' ')) == 1]
	bigrams = [i for i in feature_names if len(i.split(' ')) == 2]
	print(">",i)
	print("unigrams:",','.join(unigrams[:5]))
	print("bigrams:",','.join(bigrams[:5]))
"""

9884823387



