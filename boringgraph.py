import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm
import networkx as nx
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import csv
import sys







radid=6


#This code generates a linearly seperable graph. Adjacency matrix follows a SBM, and Feature Matrix is Gaussian. 




#Generate Features and thus classes

def generate_gaussian(num_samples, num_classes, noise):
    X = np.zeros((num_samples, 2))
    y = np.zeros(num_samples, dtype=int)
    
    samples_per_class = num_samples // num_classes

    means=np.array([[radid/2,0],[-1*radid/2,0]])

    
    for class_index in range(num_classes):
        start_index = class_index * samples_per_class
        end_index = start_index + samples_per_class
        if class_index == num_classes - 1:  
            end_index = num_samples
        
        X[start_index:end_index, 0] = np.random.normal(loc=means[class_index, 0], scale=noise, size=(end_index - start_index))
        X[start_index:end_index, 1] = np.random.normal(loc=means[class_index, 1], scale=noise, size=(end_index - start_index))
        y[start_index:end_index] = class_index
    
    return X, y


#Generate Adjacency Matrix


def generate_boring_graph(num_samples, num_classes,noise,lambdav,degree):
    X, y = generate_gaussian(num_samples=num_samples, num_classes=num_classes, noise=noise)

    
    adjacency_matrix = np.zeros((num_samples, num_samples))
    p_intra=(degree+lambdav*np.sqrt(degree))/num_samples
    p_inter =(degree-lambdav*np.sqrt(degree))/num_samples
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if y[i] == y[j]:
                if np.random.rand() < p_intra:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
            else:
                if np.random.rand() < p_inter:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    return X, adjacency_matrix, y



num_classes=2

bfeatures, badjacency, blabels =generate_boring_graph(1000,num_classes,1,3,10)
colors = ['red','cyan','magenta','blue','yellow','green']

#Visualize the graph

def visualize_graph(features, adjacency_matrix,labels):
    G = nx.from_numpy_array(adjacency_matrix)
    pos = {i: (features[i, 0], features[i, 1]) for i in range(len(features))}
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label
    plt.figure(figsize=(10, 10))
    color_map = [colors[label%4] for label in labels]
    nx.draw_networkx_nodes(G, pos, node_color=color_map, alpha=0.6, edgecolors='w')
    
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # plt.title('')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()




    # plt.figure(figsize=(5,5))
    # colors=['blue','red']
    # for class_index in range(num_classes):
    #     # Select indices belonging to the current class
    #     idx = (y == class_index)
    #     # Plot points for the current class
    #     plt.scatter(X[idx, 0], X[idx, 1], c=colors[class_index], label=f'Class {class_index}', alpha=0.5)

    # plt.title('Gaussian Graph')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


#plotting the graph
# visualize_graph(bfeatures, badjacency, blabels)

