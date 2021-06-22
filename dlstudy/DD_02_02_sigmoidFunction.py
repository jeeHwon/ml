import numpy as np
import matplotlib.pylab as plt

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
