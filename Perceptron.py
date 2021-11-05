import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T

X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
plt.show()

model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation = 'sigmoid'))
adam = Adam(lr=0.1)
model.compile(adam,loss = 'binary_crossentropy', metrics=['accuracy'])
h=model.fit(x=X, y=Y,verbose=1,epochs=500,batch_size=50, shuffle ='True')

# Plotting the accuracy
plt.plot(h.history['accuracy'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.show()

#Plotting the loss
plt.plot(h.history['loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.legend(['loss'])
plt.show()

def plot_decision_boundary(X,Y,model):
    x_span = np.linspace(min(X[:,0])-1, max(X[:,0])+1)
    y_span = np.linspace(min(X[:,1])-1,max(X[:,1])+1)
    xx,yy = np.meshgrid(x_span,y_span)
    xx_,yy_ = xx.ravel(),yy.ravel()
    print(xx_)
    print(yy_)
    grid = np.c_[xx_,yy_]
    pred_func= model.predict(grid)
    z = pred_func.reshape(xx.shape)
    print(grid)
    plt.contourf(xx,yy,z)
    plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
    plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

plot_decision_boundary(X,Y,model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
plt.show()

# Making Prediction:
plot_decision_boundary(X,Y,model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
x = 7.5
y =5
point = np.array([[x,y]])
prediction = model.predict(point)
plt.plot([x],[y],marker='o', markersize=10,color='black')
print('prediction is :', prediction)
plt.show()
