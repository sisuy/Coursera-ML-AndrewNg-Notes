import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

X=2*np.random.rand(100,1)
X=np.matrix(X)
y=4 + 3*X + np.random.rand(100,1)
y=np.matrix(y)

X_b=np.c_[np.ones((100, 1)), X]
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new=np.array([[0],[2]])
X_new_b=np.c_[np.ones((2,1)),X_new]



eta=0.1#learning rate
n_iterations=1000#迭代次数
m=100

theta=np.random.randn(2,1)

def picture(X_new,y_predict):
    plt.plot(X_new,y_predict,"r-")
    plt.plot(X,y,"b.")


for iteration in range(n_iterations):
    gradients=2/m*X_b.T.dot(X_b.dot(theta)-y)
    theta=theta-eta*gradients
    y_predict = X_new_b.dot(theta)
    picture(X_new, y_predict)

print(theta)
plt.axis([0,2,0,15])
plt.title('η=0.1')
plt.show()