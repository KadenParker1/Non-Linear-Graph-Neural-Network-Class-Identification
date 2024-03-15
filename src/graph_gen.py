"""
graph_gen.py

This file generates nonlinear graph and feature data for training on GNN's
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def generate_coords(f, num_samples, num_classes, noise,separation, mode="polar"):
    """
    Generates the positions and classes of the nodes in the graph concentrically

    Parameters:
        f (lambda): Function which determines node position
        num_samples (int): Number of nodes
        num_classes (int): Number of classes in model
        noise (float): How noisy (separated) the data is
        separation (float): How far apart the classes are
        mode (string): 'cartesian' or 'polar' depending on the domain of f

    Returns:
        pos (ndarray num_classes,): positions of nodes with one entry per class
        classes (ndarray): the class of each node
    """

    # Initialize position and class arrays
    pos = np.zeros((num_samples, 2))
    classes = np.zeros(num_samples, dtype=int)
    
    # Separate each class concentrically
    samples_per_class = num_samples // num_classes
    radii = np.linspace(1, num_classes, num_classes) 
    
    # Add each group of nodes concentrically based on class
    for class_index in range(num_classes):
        start_index = class_index * samples_per_class
        end_index = start_index + samples_per_class
        if class_index == num_classes - 1:  
            end_index = num_samples
        
        # Add nodes to class
        for i in range(start_index,end_index):
            
            # Using a polar function
            if mode == "polar":
                radius=np.random.normal(loc=separation*radii[class_index],scale=noise)
                angle=np.random.uniform(low=0,high=2*np.pi)
                pos[i, 0] = radius * f(angle) * np.cos(angle)
                pos[i, 1] = radius * f(angle) * np.sin(angle)
                classes[i] = class_index

            # Using a cartesian function
            elif mode == "cartesian":
                shift=np.random.normal(loc=separation*radii[class_index],scale=noise)
                dom=np.random.uniform(low=-5,high=5)
                pos[i, 0] = shift * dom
                pos[i, 1] = shift * f(dom)
                classes[i] = class_index
    
    # Return node features
    return pos, classes


def generate_graph(f, num_samples, num_classes,noise,lambdav,degree,separation):
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

    # Generate node positions and classes concentrically according to f
    pos, classes = generate_coords(f, num_samples=num_samples, num_classes=num_classes, noise=noise,separation=separation)
    
    # Calculate the probability of inter and intra-class connections
    adjacency_matrix = np.zeros((num_samples, num_samples))
    p_intra=(degree+lambdav*np.sqrt(degree))/num_samples
    p_inter =(degree-lambdav*np.sqrt(degree))/num_samples
    
    # Add edges according to probabilities
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if classes[i] == classes[j]:
                if np.random.rand() < p_intra:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
            else:
                if np.random.rand() < p_inter:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    # Return graph with feature data
    return pos, adjacency_matrix, classes

#Visualize the graph
colors = ['red','blue','green','yellow','orange'] # Class colors


def visualize_graph(features, adjacency_matrix, labels):
    """
    Displays the given graph using NetworkX, with distinct colors for different classes

    Parameters:
        features (ndarray num_classes,): positions of nodes with one entry per class
        adjacency_matrix (ndarray): the adjacency matrix for the graph
        labels (ndarray): the class of each node

    Returns:
        None
    """

    # Generate NewtorkX graph
    G = nx.from_numpy_array(adjacency_matrix)
    # Position-feature dictionary
    pos = {i: (features[i, 0], features[i, 1]) for i in range(len(features))}
    # Label each node
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    # Generate plot and draw
    plt.figure(figsize=(10, 10))
    color_map = [colors[label%4] for label in labels]
    nx.draw_networkx_nodes(G, pos, node_color=color_map, alpha=0.6, edgecolors='w')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()
