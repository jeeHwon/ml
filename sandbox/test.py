import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

np.random.seed(0)


num_sample = 100
minor_rate = 0.1

X = np.random.rand(num_sample*2).reshape(-1,2)
y = np.full(num_sample, 0)

minor_idx = np.random.randint(0, num_sample, size=int(num_sample*minor_rate))
for i in minor_idx:
    y[i] = 1
    
k = 3

neighbors = NearestNeighbors(n_neighbors=k).fit(X[minor_idx])

nn_array = neighbors.kneighbors([X[minor_idx][0]], k, return_distance=False)

for _ in range(10):
    nn = random.randint(1, 2)
    print(nn)