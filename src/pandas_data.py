import pandas as pd
import matplotlib.pyplot as plt


#Reads in the csvfile and plots
filename = 'experiment_results.csv'
data = pd.read_csv(filename)



plt.figure(figsize=(10, 6))
plt.plot(data['lambdav'], data['accuracy'], marker='o', linestyle='-', color='b')
plt.title('Accuracy vs Lambda Value')
plt.xlabel('Lambda Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
