import matplotlib.pyplot as plt
import numpy as np
import numpy as np
#from scipy.ndimage import convolve TODO
import pandas as pd

# Reads in the csvfile as data
filename = 'experiment_results.csv'
data = pd.read_csv(filename)

# Degree on y-axis,lambda on x-axis, accuracy as heat colors
x = np.array(data['lambdav'])
y = np.array(data['degree'])
z = np.array(data['accuracy'])

# Reshaping the data
n = len(x)
row_size = len(set(x)) # Number of unique lambdas
z = z.reshape(n//row_size,row_size)
x = x.reshape(n//row_size,row_size)
y = y.reshape(n//row_size,row_size)

# Possible TODO
# Smoothing the data with a gaussian filter
# kernel = np.array([[1,4,7,4,1],
#                    [4,16,26,16,4],
#                    [7,26,41,26,7],
#                    [4,16,26,16,4],
#                    [1,4,7,4,1]])
# z = convolve(z,kernel)/kernel.sum()
# z = convolve(z,kernel)/kernel.sum()

# Plot heatmap
plt.pcolormesh(x,y,z,vmin=0,vmax=1)
cbar = plt.colorbar()
cbar.set_label("Accuracy",rotation=270)
plt.xlabel("Lambda")
plt.ylabel("Degree")
plt.show()
