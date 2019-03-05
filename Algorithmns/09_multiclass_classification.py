from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
  
df = pd.read_csv('./flowers.csv')
X = df[list(df.columns)[:-1]]
y = df['Flower']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)  

tree = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
tree_predictions = tree.predict(X_test)   
print (tree.score(X_test, y_test))
print (confusion_matrix(y_test, tree_predictions))
print (precision_recall_fscore_support(y_test, tree_predictions))

svc = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svc_predictions = svc.predict(X_test)  
print (svc.score(X_test, y_test))
print (confusion_matrix(y_test, svc_predictions))
print (precision_recall_fscore_support(y_test, svc_predictions))

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)  
knn_predictions = knn.predict(X_test)  
print (knn.score(X_test, y_test)) 
print (confusion_matrix(y_test, knn_predictions))
print (precision_recall_fscore_support(y_test, knn_predictions))

gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
print (gnb.score(X_test, y_test))
print (confusion_matrix(y_test, gnb_predictions))
print (precision_recall_fscore_support(y_test, gnb_predictions))



நமது neural network-ல் உள்ள ஒவ்வொரு அலகுக்கும் என்னென்ன எடைகளைப் பயன்படுத்தினால், தவறுகளைக் குறைக்கலாம் எனக் கண்டுபிடிப்பதே back propagation ஆகும் . ஒவ்வொரு அடுக்கிலும் நிகழும் தவறைக் கண்டுபிடிக்க அதன் partial derivative மதிப்புகள் பின்னிருந்து முன்னாகக் கணக்கிடப்படுகின்றன. பின்னர் அவைகளை ஒன்று திரட்டி அந்த network-ன் cost கண்டுபிடிக்கப்படுகிறது. பொதுவாக gradient descent algorithm -ஆனது குறைந்த அளவு வேறுபாடு தரக்கூடிய வகையில் neuron-களின் எடையை அமைக்க இந்த back propagation -ஐப் பயன்படுத்துகிறது. 



