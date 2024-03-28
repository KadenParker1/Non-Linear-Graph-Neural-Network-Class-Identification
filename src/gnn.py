"""
gnn.py

This file contains various GNN architectures as well as the function to train them on
Graph classification problems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# Standard GCN
class GCN(torch.nn.Module):
    """
    Pytorch_Geometric implementation of GCN
    """
    def __init__(self, num_features, dimension, num_classes):
        super(GCN, self).__init__()
        #self.conv1 = lambda x,e: x@torch.ones((num_features, dimension))
        self.conv1 = GCNConv(num_features,dimension)
        self.conv2 = GCNConv(dimension, num_classes)
    
    def forward(self, data):
        """
        Runs forward propagation
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def string():
        return "GCN"

# GAT Network
class GAT(torch.nn.Module):
    """
    Pytorch_Geometric implementation of GAT
    """
    def __init__(self, in_feat, hid_feat, out_feat, log_soft = True):
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
        

def train(model,optimizer,data_loader,num_epochs,get_norms=False):
    """
    Trains the given model according to the optimizer and other specs

    Parameters:
        model (torch.nn.Module): the GNN architecture to be trained on
        optimizer (torch.optim): the training optimizer
        data_loader (DataLoader): object to load in, process and store data
        num_epochs (int): The number of epochs to train for
        get_norms (bool): Determines whether or not to compute parameter matrix norms at each epoch

    Returns:
        norms (dict, str:list): The norms for each weight and bias vector at each epoch
        (only returned if get_norms=True)
    """
    # Collect norms if get_norms is True
    if get_norms:
        norms = {name:[] for name,_ in model.conv1.named_parameters()}

    # Train on each epoch
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

        # Collect norms if needed
        if get_norms:
            for name, param in model.conv1.named_parameters():
                norms[name].append(torch.sqrt(torch.sum(param.data**2)))

        # Update on training progress
        if (epoch+1)%1000==0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')
    
    # Return norms if needed
    if get_norms:
        return norms
