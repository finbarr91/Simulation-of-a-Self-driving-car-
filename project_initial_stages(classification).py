import numpy as np
import matplotlib.pyplot as plt

n_pts = 100
np.random.seed(0) # to make sure we get the same data points each time we run the code
top_region = np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts)]).transpose()
bottom_region = np.array([np.random.normal(5,2,n_pts), np.random.normal(6,2,n_pts)]).T
_, ax= plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color = 'r')
ax.scatter(bottom_region[:, 0],bottom_region[:,1],color='b')
plt.show()

# Sigmoid Implementation.


def draw(x1, x2):
    ln = plt.plot(x1, x2)


def sigmoid(score):
    return 1 / (1 + np.exp(-score))


n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))
w1 = -0.2
w2 = -0.35
b = 3.5
line_paramters = np.matrix([w1, w2, b]).T
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -b / w2 + (x1 * (-w1 / w2))

linear_combination = all_points * line_paramters
probabilities = sigmoid(linear_combination)
print(probabilities)
_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)
plt.show()

# Cross Entropy Implementation
def draw(x1, x2):
    ln = plt.plot(x1, x2)


def sigmoid(score):
    return 1 / (1 + np.exp(-score))


def calculate_error(line_parameters, points, y):
    n = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1 / n) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy


n_pts = 10
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))
w1 = -0.1
w2 = -0.15
b = 0
line_parameters = np.matrix([w1, w2, b]).T
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -b / w2 + (x1 * (-w1 / w2))
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)
plt.show()

print((calculate_error(line_parameters, all_points, y)))

# Implementation of Gradient Descent.
def draw(x1, x2):
    ln = plt.plot(x1, x2)


def sigmoid(score):
    return 1 / (1 + np.exp(-score))


def calculate_error(line_parameters, points, y):
    n = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1 / n) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy


def gradient_descent(line_parameters, points, y, alpha):
    n = points.shape[0]
    for i in range(2000):
        p = sigmoid(points * line_parameters)
        gradient = points.T * (p - y) * (alpha / n)
        line_parameters = line_parameters - gradient

        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + (x1 * (-w1 / w2))
    draw(x1, x2)


n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

line_parameters = np.matrix([np.zeros(3)]).T
# x1=np.array([bottom_region[:,0].min(), top_region[:,0].max()])
# x2= -b/w2 + (x1*(-w1/w2))
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()