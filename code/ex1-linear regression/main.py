import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Todo:read data from cvs file,and print first 5 datas
path =  'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

#Todo:plot the graph
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#plt.show()

#Todo:Linear regression by using

#Todo:define a function to compute the loss function
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0, 'Ones', 1)

#Todo:init the data
# set X (training data) and y (target variable)
cols = data.shape[1] #cols is the column num of a matrix, data.shape[1] is the row num
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列

#Todo transform the data into matrix
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

#Todo:calculate the loss function
loss = computeCost(X, y, theta)
