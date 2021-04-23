import pandas as pd
df = pd.read_csv('data/president.csv')
print(df['height'].mean())

# numpy only
import numpy as np
data = np.loadtxt('data/president.csv', delimiter=',', skiprows=1,dtype=np.dtype)
height = data[:,2]
height = np.array(height, dtype=np.int)
print(np.mean(height))
