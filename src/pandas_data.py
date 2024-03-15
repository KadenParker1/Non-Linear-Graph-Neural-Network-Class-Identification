"""
pandas_data.py

This file generates a graph to visualize experimental results from experiments.py
"""

import pandas as pd
import matplotlib.pyplot as plt


#Reads in the csvfile and plots
filename = 'experiment_results.csv'
data = pd.read_csv(filename)


# Display data in a plot
plt.figure(figsize=(10, 6))
plt.plot(data['lambdav'], data['accuracy'], marker='o', linestyle='-', color='b')
plt.title('Accuracy vs Lambda Value')
plt.xlabel('Lambda Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
