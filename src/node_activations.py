"""
experiment.py

This file is for running various experiments with different graph structure and GNN architectures
to analyze performance
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap


# Local imports
from graph_gen import generate_graph
from gnn_activation import GCN,GAT,SAGE,train,CustomConv

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
        out,_ = model(data)

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
    
    m=CustomConv()

    # Generate model and training info
    model = arch(2,16,num_classes)
    features_tensor = torch.tensor(features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    graph_data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


    features_test, adjacency_matrix_test, labels_test = generate_graph(f,num_samples,num_classes,noise,lambdav,degree,separation)
    features_tensor2 = torch.tensor(features_test, dtype=torch.float)
    labels_tensor2 = torch.tensor(labels_test, dtype=torch.long)
    edge_index2 = torch.tensor(np.array(adjacency_matrix_test.nonzero()), dtype=torch.long)
    graph_data_test=Data(x=features_tensor2, edge_index=edge_index2, y=labels_tensor2)

    data_loader = DataLoader([graph_data], batch_size=32, shuffle=True)
    test_data = DataLoader([graph_data_test],batch_size=32,shuffle=False)
    num_epochs = epochs
    x_Agg=m.propagate(edge_index,x=features_tensor,edge_weight=None)
    print(x_Agg.shape)
    # Train the model and get test for accuracy
    activations = train(model,optimizer,data_loader,num_epochs)
    for data in test_data:
        accuracy = test(model,data)
    print(f'Node-wise Accuracy: {accuracy:.4f}')

    bias_and_weights={}
    for name,param in model.conv1.named_parameters():
        bias_and_weights[name]=param
    for i in range(len(bias_and_weights["bias"])):
        x=np.linspace(-30,30,1000)
        y=-1*(bias_and_weights["bias"][i].item()+x*bias_and_weights["lin.weight"][i,0].item())/(bias_and_weights["lin.weight"][i,1].item())
        plt.plot(x,y)
    plt.scatter(x_Agg[0],x_Agg[1])
    plt.scatter(*x_Agg.T[:,:500])
    plt.scatter(*x_Agg.T[:,500:])
    p=np.linspace(0,2*np.pi)
    plt.plot(20*np.cos(p),20*np.sin(p))
    plt.plot(10*np.cos(p),10*np.sin(p))
    plt.ylim(-30,30)
    plt.xlim(-30,30)
    plt.show()




    # Write results of training to outfile

    with open(results_file, mode='a',newline='') as file:
        writer=csv.writer(file)
        writer.writerow([accuracy,lambdav,separation])


    scaling=StandardScaler()
    data=activations[num_epochs-1]

    # data=m.propagate(edge_index,x=torch.Tensor(data),edge_weight=None).numpy()
    
   
    scaled_data=scaling.fit_transform(data)
    
    pca=umap.UMAP(n_components=3)
    # pca=PCA(n_components=3)
    x=pca.fit_transform(scaled_data)

    if x.shape[1]==2:
        plt.scatter(x[:num_samples//num_classes,0],x[:num_samples//num_classes,1])
        plt.scatter(x[num_samples//num_classes:,0],x[num_samples//num_classes:,1])
        plt.show()

    # 3d graphing
    if x.shape[1]==3:
        fig=plt.figure()
        ax=fig.add_subplot(projection='3d')
        ax.scatter(x[:num_samples//num_classes,0],x[:num_samples//num_classes,1],x[:num_samples//num_classes,2])
        ax.scatter(x[num_samples//num_classes:,0],x[num_samples//num_classes:,1],x[num_samples//num_classes:,2])

        #for more than 2 classes
        if num_classes==3:
            ax.scatter(x[2*num_samples//num_classes:,0],x[2*num_samples//num_classes:,1],x[2*num_samples//num_classes:,2])

        if num_classes==4:
            ax.scatter(x[2*num_samples//num_classes:,0],x[2*num_samples//num_classes:,1],x[2*num_samples//num_classes:,2],edgecolor='cyan')
            ax.scatter(x[3*num_samples//num_classes:,0],x[3*num_samples//num_classes:,1],x[3*num_samples//num_classes:,2],edgecolor='purple')

        plt.show()

    # if you want to see it
    # print(pca.explained_variance_ratio_)



    # Return the nodewise accuracy
    return accuracy,activations

# def aggregate_norms(num_runs=10):
#     """
#     Aggregates the norms over a number of runs
#     """
#     all_norms = []

#     for i in range(num_runs):
#         print(i)
#         norms = run_experiment(5000,1,1000,num_classes=2,lambdav=0,degree=10,separation=.5,arch=SAGE)
#         all_norms.append(norms)
#     avg_norms = {name: np.mean([run[name] for run in all_norms], axis=0) for name in all_norms[0]}
#     return avg_norms

# def plot_avg_norms(avg_norms):
#     """
#     Plots the average norm over a number of runs
#     """
#     for key in avg_norms:
#         plt.plot(avg_norms[key], label=key)
#     plt.legend()
#     plt.title("Average Norms")
#     plt.xlabel("Epoch")
#     plt.ylabel("Norm")
#     plt.show()

# Run some tests, if desired
if __name__ == "__main__":
    archs = [GCN]
    for arch in archs:
        print(arch.string())
        run_experiment(2000,.01,1000,num_classes=2,lambdav=3,degree=10,separation=5,arch=arch)

    # Analyze norms
    # avg_norms = aggregate_norms(num_runs=5)
    # plot_avg_norms(avg_norms)


# Do multiple tests and write to CSV (not recommended unless on supercomputer)
    
# def fill_it_out():
#     for 1000*samples in paramters['num_samples']:
#         accuracy=run_experiment(epochs,lr,noise,samples,classes,homophily,heterophily)
#         writer.writerow([epochs, lr, noise, samples, classes, accuracy])
#         print(f'Done: epochs={epochs}, lr={lr}, noise={noise}, samples={samples}, classes={classes}, accuracy={accuracy}')

