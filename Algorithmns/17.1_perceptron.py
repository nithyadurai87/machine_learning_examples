def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation > 0.0 else 0.0	
	
def train_weights(dataset, l_rate, n_epoch):
    weights = [0.0 for i in range(len(dataset[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in dataset:
            error = row[-1] - predict(row, weights)
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('epoch=%d, error=%.2f' % (epoch, sum_error))
    print (weights)
 
dataset = [[0.4,0.3,1],
	[0.6,0.8,1],
	[0.7,0.5,1],
	[0.9,0.2,0]]
	
l_rate = 0.1
n_epoch = 6
train_weights(dataset, l_rate, n_epoch)

"""
http://marubon-ds.blogspot.com/2017/06/title.html
https://stats.stackexchange.com/questions/281623/knn-outperforms-cnn
https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/?completed=/convolutional-neural-network-cnn-machine-learning-tutorial/


https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/






"""

