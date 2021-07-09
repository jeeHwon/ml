import numpy as np
import matplotlib.pyplot as plt
from smote_basic import Smote, plot_2d_space

np.random.seed(0)

num_sample = 100
minor_rate = 0.1

X = np.random.rand(num_sample*2).reshape(-1,2)
y = np.full(num_sample, 0)

minor_idx = np.random.randint(0, num_sample, size=int(num_sample*minor_rate))
for i in minor_idx:
    y[i] = 1


s = Smote(sample=X[minor_idx], N=300, k=3)
s.over_sampling()

X_smp = np.concatenate((X, s.synthetic), axis=0)
new_label = np.full(len(s.synthetic), 2)
y_smp = np.concatenate((y, new_label), axis=0)


# plot_2d_space(X_smp, y_smp, 'test')

import pandas as pd

df = pd.DataFrame(X_smp, y_smp)
print(df)