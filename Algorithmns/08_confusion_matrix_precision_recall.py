from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

print ('Accuracy:', accuracy_score(y_true, y_pred))
print (confusion_matrix(y_pred,y_true))
print (precision_recall_fscore_support(y_true, y_pred))

plt.matshow(confusion_matrix(y_true, y_pred))
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


