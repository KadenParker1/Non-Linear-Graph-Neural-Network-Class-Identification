import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm
import networkx as nx
from torch_geometric.data import DataLoader


radid=6


#This code generates a linearly seperable graph. Adjacency matrix follows a SBM, and Feature Matrix is Gaussian. 


def generate_gaussian(num_samples, num_classes, noise):
    """
    Generates Gaussian features and classes

    Parameters:
        num_samples (int): Number of nodes
        num_classes (int): Number of classes in model
        noise (float): How noisy (separated) the data is

    Returns:
        pos (ndarray num_classes,): positions of nodes with one entry per class
        classes (ndarray): the class of each node
    """

    # Initialize position and class arrays
    pos = np.zeros((num_samples, 2))
    classes = np.zeros(num_samples, dtype=int)
    
    samples_per_class = num_samples // num_classes

    # Class means (centers) for Gaussian clouds
    means=np.array([[radid/2,0],[-1*radid/2,0]])

    # Place nodes in Gaussian clouds around the means
    for class_index in range(num_classes):
        start_index = class_index * samples_per_class
        end_index = start_index + samples_per_class
        if class_index == num_classes - 1:  
            end_index = num_samples
        
        pos[start_index:end_index, 0] = np.random.normal(loc=means[class_index, 0], scale=noise, size=(end_index - start_index))
        pos[start_index:end_index, 1] = np.random.normal(loc=means[class_index, 1], scale=noise, size=(end_index - start_index))
        classes[start_index:end_index] = class_index
    
    # Return results
    return pos, classes


def generate_boring_graph(num_samples, num_classes,noise,lambdav,degree):
    """
    Generates adjacency matrix for the graph by placing edges between nodes

    Parameters:
        num_samples (int): Number of nodes
        num_classes (int): Number of classes in model
        noise (float): How noisy (separated) the data is
        lambdav (float): Dictates the level of homophily or heterophily in the graph
        degree (float): Average degree of the graph

    Returns:
        pos (ndarray num_classes,): positions of nodes with one entry per class
        adjacency_matrix (ndarray): the adjacency matrix for the graph
        classes (ndarray): the class of each node
    """

    # Generate node positions in Gaussian clouds
    pos, classes = generate_gaussian(num_samples=num_samples, num_classes=num_classes, noise=noise)

    # Initialize variables
    adjacency_matrix = np.zeros((num_samples, num_samples))
    p_intra=(degree+lambdav*np.sqrt(degree))/num_samples # Intra-edge probability
    p_inter =(degree-lambdav*np.sqrt(degree))/num_samples # Inter-edge probability
    
    # Add edges according to probabilities
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if classes[i] == classes[j]:
                if np.random.rand() < p_intra:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
            else:
                if np.random.rand() < p_inter:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    # Return graph alongside feature information
    return pos, adjacency_matrix, classes



num_classes=2

# Generate graph with features
features, adjacency, labels =generate_boring_graph(1000,num_classes,1,3,10)
colors = ['red','cyan','magenta','blue','yellow','green'] # Colors for visualizing classes

#Visualize the graph
def visualize_graph(features, adjacency_matrix,labels):
    """
    Displays the given graph using NetworkX, with distinct colors for different classes

    Parameters:
        features (ndarray num_classes,): positions of nodes with one entry per class
        adjacency_matrix (ndarray): the adjacency matrix for the graph
        labels (ndarray): the class of each node

    Returns:
        None
    """
    # Generate nx graph
    G = nx.from_numpy_array(adjacency_matrix)
    # Position-feature dictionary
    pos = {i: (features[i, 0], features[i, 1]) for i in range(len(features))}
    # Label each node
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    # Generate the plot and draw
    plt.figure(figsize=(10, 10))
    color_map = [colors[label%4] for label in labels]
    nx.draw_networkx_nodes(G, pos, node_color=color_map, alpha=0.6, edgecolors='w')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()


# plotting the graph
#visualize_graph(bfeatures, badjacency, blabels)

