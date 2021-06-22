import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-2, 2.5, 0.25)
y = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x, y) # 그물망 만들기
Z = X**2 + Y**2

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
plt.show()
plt.contour(X,Y,Z) # 등위선
plt.show()

