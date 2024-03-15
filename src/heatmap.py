"""
heatmap.py

This file generates a heatmap form csv files found in test_runs
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
import pandas as pd

# Reads in the csvfile as data
filename = '../test_runs/experiment_results.csv'
data = pd.read_csv(filename)

# Degree on y-axis,lambda on x-axis, accuracy as heat colors
x = np.array(data['lambdav'])
y = np.array(data['radid'])
z = np.array(data['accuracy'])
n = len(x)
row_size = len(set(x))

# Reshaping the data
z = z.reshape(n//row_size,row_size)
x = x.reshape(n//row_size,row_size)
y = y.reshape(n//row_size,row_size)

# Smoothing the data with a gaussian filter
# 3 x 3
# kernel = np.array([[1,2,1],
#                    [2,4,2],
#                    [1,2,1]])

# 5 x 5
kernel = np.array([[1,4,7,4,1],
                   [4,16,26,16,4],
                   [7,26,41,26,7],
                   [4,16,26,16,4],
                   [1,4,7,4,1]])

# 7 x 7
# kernel = np.array([[0,0,1,2,1,0,0],
#                    [0,3,13,22,13,3,0],
#                    [1,13,59,97,59,13,1],
#                    [2,22,97,159,97,22,2],
#                    [1,13,59,97,59,13,1],
#                    [0,3,13,22,13,3,0],
#                    [0,0,1,2,1,0,0],])

# Apply the kernel smoothing
z = convolve(z,kernel)/kernel.sum()
z = convolve(z,kernel)/kernel.sum()

# Plot heatmap
plt.pcolormesh(x,y,z,vmin=.5,vmax=1,cmap="coolwarm")
cbar = plt.colorbar()
cbar.set_label("Accuracy",rotation=270)
plt.xlabel("Lambda")
plt.ylabel("Radius")
plt.title("Nodewise Accuracy")
plt.yticks(np.arange(0,11))
plt.xticks(np.linspace(-3,3,16),rotation=90)
plt.show()
