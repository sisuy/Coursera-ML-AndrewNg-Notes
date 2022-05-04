import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Todo:read data from cvs file,and print first 5 datas
path =  'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

#Todo:plot the graph
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

plt.show()

#size = data.iloc[:,cols - 1]


#Todo:Linear regression by using
#Todo:define a function to compute the loss function
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0, 'Ones', 1)

#Todo:init the dataw
# set X (training data) and y (target variable)
m = data.shape[0]
cols = data.shape[1] #cols is the column num of a matrix, data.shape[1] is the row num
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列

#Todo transform the data into matrix
X = np.matrix(X.values)
y = np.matrix(y.values)

#init theta [0,0]
theta = np.matrix(np.array([0,0]))

#Todo:calculate the loss function
loss = computeCost(X, y, theta)

#Todo:gradient Descent to minmize the cost function
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T) - y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(X))*np.sum(term))
            theta = temp
            cost[i] = computeCost(X, y, theta)

    return theta, cost

alpha = 0.001
iters = 50000

g,cost = gradientDescent(X,y,theta,alpha,iters)
g

computeCost(X,y,g)

#Todo:plot the final graph
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))

ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

#Todo:plot cost function
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()