"""
import nltk
nltk.download()
"""
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

def lemmatize(token, tag):
	if tag[0].lower() in ['n', 'v']:
		return WordNetLemmatizer().lemmatize(token, tag[0].lower())
	return token
	
corpus = ['Bird is a Peacock Bird','Peacock dances very well','It eats variety of seeds','Cumin seed was eaten by it once']

print (CountVectorizer().fit_transform(corpus).todense())
print (CountVectorizer(stop_words='english').fit_transform(corpus).todense())

print (PorterStemmer().stem('seeds'))

print (WordNetLemmatizer().lemmatize('gathering', 'v'))
print (WordNetLemmatizer().lemmatize('gathering', 'n'))

s_lines=[]
for document in corpus:
	s_words=[]
	for token in word_tokenize(document):
		s_words.append(PorterStemmer().stem(token))
	s_lines.append(s_words)
print ('Stemmed:',s_lines)

tagged_corpus=[]
for document in corpus:
	tagged_corpus.append(pos_tag(word_tokenize(document)))

l_lines=[]
for document in tagged_corpus:
	l_words=[]
	for token, tag in document:
		l_words.append(lemmatize(token, tag))
	l_lines.append(l_words)
print ('Lemmatized:',l_lines)

"""
[[2 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1]
 [0 0 0 0 0 1 0 1 1 0 0 0 1 1 0 0 0]
 [0 1 1 0 1 0 0 1 0 1 0 1 0 0 0 1 0]]
 
words like is, very, well, it, of, was, by, once are ignored. hence 4*9
[[2 0 0 0 0 1 0 0 0]
 [0 0 1 0 0 1 0 0 0]
 [0 0 0 0 1 0 0 1 1]
 [0 1 0 1 0 0 1 0 0]]
 
you could still see that seed and seeds are separate words
print (PorterStemmer().stem('seeds'))
seed 

print (WordNetLemmatizer().lemmatize('gathering', 'v'))
gather

print (WordNetLemmatizer().lemmatize('gathering', 'n'))
gathering

print ('Stemmed:',s_lines)
Stemmed: [['bird', 'is', 'a', 'peacock', 'bird'], ['peacock', 'danc', 'veri', 'w
ell'], ['It', 'eat', 'varieti', 'of', 'seed'], ['cumin', 'seed', 'wa', 'eaten',
'by', 'it', 'onc']]

print ('Lemmatized:',l_lines)
Lemmatized: [['Bird', 'be', 'a', 'Peacock', 'Bird'], ['Peacock', 'dance', 'very', 'well'],
['It', 'eat', 'variety', 'of', 'seed'], ['Cumin', 'seed', 'be', 'eat', 'by', 'it', 'once']]

"""
