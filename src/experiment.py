"""
experiment.py

This file is for running various experiments with different graph structure and GNN architectures
to analyze performance
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import csv

# Local imports
from graph_gen import generate_graph
from gnn import GCN,GAT,SAGE,train

#Making a csv file to record the desired parameters
results_file= '../test_runs/experiment_results.csv'
with open(results_file, mode='w',newline='') as file:
    writer=csv.writer(file)
    writer.writerow(['accuracy','lambdav','noise','num_samples','num_classes','epochs','degree','separation'])


def test(model,data):
    """
    Tests trained network to determine nodewise classification accuracy
    """
    
    # Obtain and evaluate model data
    model.eval()
    with torch.no_grad():  
        out = model(data)

        # Compute percent correct predictions        
        _, pred = out.max(dim=1)  
        correct = pred.eq(data.y).sum().item()  
        acc = correct / data.num_nodes  

    return acc

def run_experiment(epochs,noise,num_samples,num_classes,lambdav,degree,separation,arch=GCN):
    """
    Generates a graph with SBM data and trains a given GNN architecture to determine nodewise accuracy

    Parameters:
        epochs (int): number of epochs to train for
        noise (float): how noisy (separated) the data is
        num_samples (int): number of nodes
        num_classes (int): number of classes in model
        lambdav (float): dictates the level of homophily or heterophily in the graph
        degree (float): average degree of the graph
        separation (float): how far apart the classes are
        arch (torch.nn.Module): the GNN architecture to be trained on


    Returns:
        accuracy (float): the nodewise classification accuracy of the GNN on the graph
        norms (dict str:list): the norms of network parameters at each epoch
    """

    # This function dictates the shape of the node data
    f = lambda t: 1

    # generate graph and feature data
    features, adjacency_matrix, labels = generate_graph(f,num_samples,num_classes,noise,lambdav,degree,separation)
    
    # Generate model and training info
    model = arch(2,16,num_classes)
    features_tensor = torch.tensor(features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    graph_data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    data_loader = DataLoader([graph_data], batch_size=32, shuffle=True)
    num_epochs = epochs

    # Train the model and get accuracy
    norms = train(model,optimizer,data_loader,num_epochs)
    accuracy = test(model,graph_data)
    print(f'Node-wise Accuracy: {accuracy:.4f}')

    # Write results of training to outfile
    with open(results_file, mode='a',newline='') as file:
        writer=csv.writer(file)
        writer.writerow([accuracy,lambdav,separation])

    # Return the nodewise accuracy
    return accuracy,norms

def aggregate_norms(num_runs=10):
    """
    Aggregates the norms over a number of runs
    """
    all_norms = []

    for i in range(num_runs):
        print(i)
        norms = run_experiment(5000,1,1000,num_classes=2,lambdav=0,degree=10,separation=.5,arch=SAGE)
        all_norms.append(norms)
    avg_norms = {name: np.mean([run[name] for run in all_norms], axis=0) for name in all_norms[0]}
    return avg_norms

def plot_avg_norms(avg_norms):
    """
    Plots the average norm over a number of runs
    """
    for key in avg_norms:
        plt.plot(avg_norms[key], label=key)
    plt.legend()
    plt.title("Average Norms")
    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    plt.show()

# Run some tests, if desired
if __name__ == "__main__":
    archs = [SAGE]
    for arch in archs:
        print(arch.string())
        run_experiment(5000,0,1000,num_classes=2,lambdav=3,degree=10,separation=5,arch=arch)

    # Analyze norms
    # avg_norms = aggregate_norms(num_runs=5)
    # plot_avg_norms(avg_norms)


# Do multiple tests and write to CSV (not recommended unless on supercomputer)
    
# def fill_it_out():
#     for 1000*samples in paramters['num_samples']:
#         accuracy=run_experiment(epochs,lr,noise,samples,classes,homophily,heterophily)
#         writer.writerow([epochs, lr, noise, samples, classes, accuracy])
#         print(f'Done: epochs={epochs}, lr={lr}, noise={noise}, samples={samples}, classes={classes}, accuracy={accuracy}')

