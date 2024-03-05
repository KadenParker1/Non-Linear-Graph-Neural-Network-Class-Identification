import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import DataLoader
import csv

def generate_coords(f, num_samples, num_classes, noise,separation, mode="polar"):
    """
    Generates the positions and classes of the nodes in the graph

    Parameters:
        f (lambda): Function which determines node position
        num_samples (int): Number of nodes
        num_classes (int): Number of classes in model
        noise (float): How noisy (separated) the data is
        separation (float): How far apart the classes are
        mode (string): 'cartesian' or 'polar' depending on the domain of f

    Returns:
        X (ndarray num_classes,): positions of nodes with one entry per class
        y (ndarray): the class of each node
    """

    X = np.zeros((num_samples, 2))
    y = np.zeros(num_samples, dtype=int)
    
    samples_per_class = num_samples // num_classes
    radii = np.linspace(1, num_classes, num_classes) 
    
    for class_index in range(num_classes):
        start_index = class_index * samples_per_class
        end_index = start_index + samples_per_class
        if class_index == num_classes - 1:  
            end_index = num_samples
        
        for i in range(start_index,end_index):

            if mode == "polar":
                radius=np.random.normal(loc=separation*radii[class_index],scale=noise)
                angle=np.random.uniform(low=0,high=2*np.pi)
                X[i, 0] = radius * f(angle) * np.cos(angle)
                X[i, 1] = radius * f(angle) * np.sin(angle)
                y[i] = class_index

            elif mode == "cartesian":
                shift=np.random.normal(loc=separation*radii[class_index],scale=noise)
                dom=np.random.uniform(low=-5,high=5)
                X[i, 0] = shift * dom
                X[i, 1] = shift * f(dom)
                y[i] = class_index
    
    return X, y


def generate_graph(f, num_samples, num_classes,noise,lambdav,degree,separation):
    X, y = generate_coords(f, num_samples=num_samples, num_classes=num_classes, noise=noise,separation=separation)

#Generate adjacency matrix SBM

    
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

#Visualize the graph
colors = ['red','blue','green','yellow','orange']
def visualize_graph(features, adjacency_matrix, labels):
    G = nx.from_numpy_array(adjacency_matrix)
    pos = {i: (features[i, 0], features[i, 1]) for i in range(len(features))}
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label
    plt.figure(figsize=(10, 10))
    color_map = [colors[label%4] for label in labels]
    nx.draw_networkx_nodes(G, pos, node_color=color_map, alpha=0.6, edgecolors='w')
    
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()


# GNN Architectures
    
# Standard GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features,dimension, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, dimension)
        self.conv2 = GCNConv(dimension, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def string():
        return "GCN"



#Defining GNN tasked with identifying the class for a given node 

class GAT(torch.nn.Module):
    """
    Pytorch_Geometric implementation of GAT

    """
    def __init__(self, in_feat, hid_feat, out_feat, log_soft = True):
        """
        Constructor of class
        """
        super().__init__()
        self.conv1 = GATConv(in_feat, hid_feat)
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = GATConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        self.log_soft = log_soft
        #self.dropout = nn.Dropout(p=.4)
    def forward(self, data):
        """
        Runs forward propagation
        """
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        if self.log_soft is True:
            return F.log_softmax(x,dim=1)
        else:
            return x
    def string():
        return "GAT"

# Graph Sage with neighborhood feature aggregation
class SAGE(torch.nn.Module):
    """
    Pytorch_Geometric implementation of SAGE
    """
    def __init__(self, in_feat, hid_feat, out_feat, log_soft = True):
        """
        Constructor of class
        """
        super().__init__()
        self.conv1 = SAGEConv(in_feat, hid_feat)
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = SAGEConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        self.log_soft = log_soft
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, data):
        """
        Runs forward propagation
        """
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        if self.log_soft is True:
            return F.log_softmax(x,dim=1)
        else:
            return x
    def string():
        return "SAGE"
        

#Train the GNN

def train(model,optimizer,data_loader,num_epochs):

    norms = {name:[] for name,_ in model.conv1.named_parameters()}
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_nodes  
        
        total_loss /= len(data_loader.dataset)
        for name, param in model.conv1.named_parameters():
            norms[name].append(torch.norm(param.data,p=2))

        # if (epoch+1)%1000==0:
            # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')
    return norms


#Tests to find the nodewise accuracy


def test(model,data):
    model.eval()
    with torch.no_grad():  
        out = model(data)
        # print(out.shape)
        _, pred = out.max(dim=1)  
        correct = pred.eq(data.y).sum().item()  
        acc = correct / data.num_nodes  
    return acc


#Making a csv file to record the desired parameters

results_file= '../test_runs/experiment_results.csv'
with open(results_file, mode='w',newline='') as file:
    writer=csv.writer(file)

    writer.writerow(['accuracy','lambdav','noise','num_samples','num_classes','epochs','degree','separation'])


def run_experiment(epochs,noise,num_samples,num_classes,lambdav,degree,separation,arch=GCN):
    f = lambda t: 1+2*t
    # features,adjacency_matrix,labels=bfeatures, badjacency, blabels
    features, adjacency_matrix, labels = generate_graph(f,num_samples,num_classes,noise,lambdav,degree,separation)
    # model = GCN(num_features=2, num_classes=num_classes)
    model = arch(2,16,num_classes)
    features_tensor = torch.tensor(features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    graph_data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    data_loader = DataLoader([graph_data], batch_size=32, shuffle=True)
    num_epochs = epochs
    norms = train(model,optimizer,data_loader,num_epochs)
    accuracy = test(model,graph_data)
    print(f'Node-wise Accuracy: {accuracy:.4f}')
    with open(results_file, mode='a',newline='') as file:
        writer=csv.writer(file)
        writer.writerow([accuracy,lambdav,separation])
    
    # Plotting norms
    # for key in norms:
    #     plt.plot(norms[key],label=key)
    # plt.legend()
    # plt.show()

    # for name,param in model.conv1.named_parameters():
    #     print(param.data)
    
    return norms

# Run some tests, if desired
# if __name__ == "__main__":
#     archs = [SAGE]
#     for arch in archs:
#         run_experiment(3000,1,1000,num_classes=2,lambdav=3,degree=10,separation=10,arch=arch)


# # Do multiple tests and write to CSV (not recommended unless on supercomputer)
    
# def fill_it_out():
#     for 1000*samples in paramters['num_samples']:
#         accuracy=run_experiment(epochs,lr,noise,samples,classes,homophily,heterophily)
#         writer.writerow([epochs, lr, noise, samples, classes, accuracy])
#         print(f'Done: epochs={epochs}, lr={lr}, noise={noise}, samples={samples}, classes={classes}, accuracy={accuracy}')
        
def aggregate_norms(num_runs=10):
    all_norms = []

    for i in range(num_runs):
        print(i)
        norms = run_experiment(5000,1,1000,num_classes=2,lambdav=0,degree=10,separation=1,arch=SAGE)
        all_norms.append(norms)
    avg_norms = {name: np.mean([run[name] for run in all_norms], axis=0) for name in all_norms[0]}
    return avg_norms

def plot_avg_norms(avg_norms):
    for key in avg_norms:
        plt.plot(avg_norms[key], label=key)
    plt.legend()
    plt.title("Average Norms")
    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    plt.show()

avg_norms = aggregate_norms(num_runs=10)
plot_avg_norms(avg_norms)