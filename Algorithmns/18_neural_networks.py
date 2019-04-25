from mlxtend.classifier import MultiLayerPerceptron as MLP
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np 

X = np.asarray([[6.1,1.4],[7.7,2.3],[6.3,2.4],[6.4,1.8],[6.2,1.8],[6.9,2.1],
[6.7,2.4],[6.9,2.3],[5.8,1.9],[6.8,2.3],[6.7,2.5],[6.7,2.3],[6.3,1.9],[6.5,2.1 ],[6.2,2.3],[5.9,1.8]] ) 

X = (X - X.mean(axis=0)) / X.std(axis=0)

y = np.asarray([0,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2]) 

nn = MLP(hidden_layers=[50],l2=0.00,l1=0.0,epochs=150,eta=0.05,
momentum=0.1,decrease_const=0.0,minibatches=1,random_seed=1,print_progress=3)
nn = nn.fit(X, y)

fig = plot_decision_regions(X=X, y=y, clf=nn, legend=2)
plt.show()
print('Accuracy(epochs = 150): %.2f%%' % (100 * nn.score(X, y)))

nn.epochs = 250
nn = nn.fit(X, y)
fig = plot_decision_regions(X=X, y=y, clf=nn, legend=2)
plt.title('epochs = 250')
plt.show()
print('Accuracy(epochs = 250): %.2f%%' % (100 * nn.score(X, y)))

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.title('Gradient Descent training (minibatches=1)')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

nn.minibatches = len(y)
nn = nn.fit(X, y)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.title('Stochastic Gradient Descent (minibatches=no. of training examples)')
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

#http://rasbt.github.io/mlxtend/user_guide/classifier/MultiLayerPerceptron/




