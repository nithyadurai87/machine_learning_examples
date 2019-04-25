from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus1 = [{'Gender': 'Male'},{'Gender': 'Female'},{'Gender': 'Transgender'},{'Gender': 'Male'},{'Gender': 'Female'}]
corpus2 = ['Bird is a Peacock Bird','Peacock dances very well','It eats variety of seeds','Cumin seed was eaten by it once']
vectors = [[2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],[0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]]
 
# one-hot encoding
v1 = DictVectorizer()
print (v1.fit_transform(corpus1).toarray())
print (v1.vocabulary_)

# bag-of-words (term frequencies, binary frequencies)
v2 = CountVectorizer()
print (v2.fit_transform(corpus2).todense())
print (v2.vocabulary_)

print (TfidfVectorizer().fit_transform(corpus2).todense())

print (HashingVectorizer(n_features=6).transform(corpus2).todense())

print (euclidean_distances([vectors[0]],[vectors[1]]))
print (euclidean_distances([vectors[0]],[vectors[2]]))
print (euclidean_distances([vectors[0]],[vectors[3]]))

"""

https://gist.github.com/nithyadurai87/f3fff58ab7272279ef069689fc391dec

https://gist.github.com/nithyadurai87/491e5e6f9c009ebd88912e71ef9363a4


print (v1.fit_transform(corpus1).toarray())
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]

print (v1.vocabulary_)
{'Gender=Male': 1, 'Gender=Female': 0, 'Gender=Transgender': 2}

print (v2.fit_transform(corpus2).todense())
[[2 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1]
 [0 0 0 0 0 1 0 1 1 0 0 0 1 1 0 0 0]
 [0 1 1 0 1 0 0 1 0 1 0 1 0 0 0 1 0]]

print (v2.vocabulary_)
{'bird': 0, 'is': 6, 'peacock': 10, 'dances': 3, 'very': 14, 'well': 16, 'it': 7
, 'eats': 5, 'variety': 13, 'of': 8, 'seeds': 12, 'cumin': 2, 'seed': 11, 'was':
 15, 'eaten': 4, 'by': 1, 'once': 9}

print (TfidfVectorizer().fit_transform(corpus2).todense())
[[0.84352956 0.         0.         0.         0.         0.
  0.42176478 0.         0.         0.         0.3325242  0.
  0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.52547275 0.         0.
  0.         0.         0.         0.         0.41428875 0.
  0.         0.         0.52547275 0.         0.52547275]
 [0.         0.         0.         0.         0.         0.46516193
  0.         0.36673901 0.46516193 0.         0.         0.
  0.46516193 0.46516193 0.         0.         0.        ]
 [0.         0.38861429 0.38861429 0.         0.38861429 0.
  0.         0.30638797 0.         0.38861429 0.         0.38861429
  0.         0.         0.         0.38861429 0.        ]]  

print (HashingVectorizer(n_features=6).transform(corpus2).todense())
[[ 0.         -0.70710678 -0.70710678  0.          0.          0.        ]
 [ 0.          0.         -0.81649658 -0.40824829  0.40824829  0.        ]
 [ 0.75592895  0.         -0.37796447  0.         -0.37796447 -0.37796447]
 [ 0.25819889  0.77459667  0.         -0.51639778  0.          0.25819889]]

print (euclidean_distances([vectors[0]],[vectors[1]]))
[[2.82842712]]

print (euclidean_distances([vectors[0]],[vectors[2]]))
[[3.31662479]]

print (euclidean_distances([vectors[0]],[vectors[3]]))
[[3.60555128]]
"""



