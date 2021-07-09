from sklearn.neighbors import NearestNeighbors
import random
import numpy as np
from matplotlib import pyplot as plt
 
class Smote:
   def __init__(self, sample, N, k):
       self.sample = sample 
       self.k = k + 1 # +1 추가 / 자기 자신빼고 진짜 knn 하기 위해
       self.T = len(self.sample)
       self.N = N
       self.newIndex = 0
       self.synthetic = []
       self.neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.sample)
 
   def over_sampling(self):
       if self.N < 100:
           self.T = (self.N / 100) * self.T
           self.N = 100
       self.N = int(self.N / 100)
 
       for i in range(0, self.T):
           nn_array = self.compute_k_nearest(i)
           self.populate(self.N, i, nn_array)
 
   def compute_k_nearest(self, i):
       nn_array = self.neighbors.kneighbors([self.sample[i]], self.k, return_distance=False)
       if len(nn_array) == 1:
           return nn_array[0]
       else:
           return []
        
  
   def populate(self, N, i, nn_array):
       while N != 0:
           nn = random.randint(1, self.k-1) # 오류 발견 0이 아니라 1이어야함(자신 제외)
           self.synthetic.append([])
           for attr in range(0, len(self.sample[i])):
               dif = self.sample[nn_array[nn]][attr] - self.sample[i][attr]
               gap = random.random()
               self.synthetic[self.newIndex].append(self.sample[i][attr] + gap * dif)
           self.newIndex += 1
           N -= 1


def plot_2d_space(X, y, label='Classes'):  
   colors = ['#1F77B4', 'green', 'red']
   markers = ['o', 's', 'o']
   for l, c, m in zip(np.unique(y), colors, markers):
       plt.scatter(
           X[y==l, 0],
           X[y==l, 1],
           c=c, label=l, marker=m, alpha=0.5
       )
   plt.title(label)
   plt.legend(loc='upper right')
   plt.show()

# test
'''
X = np.array(
   [
       [0.11622591, -0.0317206],
       [0.77481731, 0.60935141],
       [1.25192108, -0.22367336],
       [0.53366841, -0.30312976],
       [1.52091956, -0.49283504],
       [-0.28162401, -2.10400981],
       [0.83680821, 1.72827342],
       [0.3084254, 0.33299982],
       [0.70472253, -0.73309052],
       [0.28893132, -0.38761769],
       [1.15514042, 0.0129463],
       [0.88407872, 0.35454207],
       [1.31301027, -0.92648734],
       [-1.11515198, -0.93689695],
       [-0.18410027, -0.45194484],
       [0.9281014, 0.53085498],
       [-0.14374509, 0.27370049],
       [-0.41635887, -0.38299653],
       [0.08711622, 0.93259929],
       [1.70580611, -0.11219234],
   ]
)
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
 
# print(s.synthetic)
 
X_minor = X[:4, :]
 
s = Smote(sample=X_minor, N=300, k=3)
s.over_sampling()
 
 
X_smp = np.concatenate((X, s.synthetic), axis=0)
new_label = np.full(len(s.synthetic), 2)
y_smp = np.concatenate((y, new_label), axis=0)
 
print(X_smp)
print(y_smp)
 
plot_2d_space(X_smp, y_smp, 'test')
'''