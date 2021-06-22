import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=np.int)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y1 = step_function(X)
Y2 = sigmoid_function(X)
plt.plot(X, Y1)
plt.plot(X, Y2, 'r--')
plt.ylim(-0.1, 1.1)
plt.show()
