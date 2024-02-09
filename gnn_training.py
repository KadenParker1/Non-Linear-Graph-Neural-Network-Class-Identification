import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm
import networkx as nx
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader



def generate_points(num_samples, num_classes, noise, f=lambda x:1, mode="polar"):
    X = np.zeros((num_samples, 2))
    y = np.zeros(num_samples, dtype=int)
    
    samples_per_class = num_samples // num_classes
    radii = np.linspace(1, num_classes, num_classes) 
    
    for class_index in range(num_classes):
        start_index = class_index * samples_per_class
        end_index = start_index + samples_per_class
        if class_index == num_classes - 1:  
            end_index = num_samples
        n = end_index-start_index
        shift=np.random.normal(loc=radii[class_index],scale=noise,size=n)
        dom=np.random.uniform(low=-5,high=5,size=n)
        if mode == "polar":
            X[start_index:end_index, 0] = shift * f(dom) * np.cos(dom)
            X[start_index:end_index, 1] = shift * f(dom) * np.sin(dom)
            y[start_index:end_index] = class_index
        elif mode == "cartesian":
            X[start_index:end_index, 0] = shift * dom
            X[start_index:end_index, 1] = shift * f(dom)
            y[start_index:end_index] = class_index
        else:
            raise ValueError("Mode must be either 'polar' or 'cartesian'")
        # for i in range(start_index,end_index):
        #     shift=np.random.normal(loc=radii[class_index],scale=noise)
        #     dom=np.random.uniform(low=-5,high=5)
        #     X[i, 0] = shift * dom
        #     X[i, 1] = shift * f(dom)
        #     y[i] = class_index
        # angles = np.random.uniform(low=0, high=2*np.pi, size=(end_index - start_index,))
        # radii_noise = np.random.normal(loc=radii[class_index], scale=noise, size=(end_index - start_index,))
        # X[start_index:end_index, 0] = radii_noise * np.cos(angles)
        # X[start_index:end_index, 1] = radii_noise * np.sin(angles)
        # y[start_index:end_index] = class_index
    
    return X, y





def generate_graph(num_samples, num_classes,noise,f=lambda x:1, mode="polar"):
    X, y = generate_points(num_samples, num_classes, noise, f, mode)

    # Generate SBM-like edges within each class
    adjacency_matrix = np.zeros((num_samples, num_samples))
    p_intra, p_inter = 0.01, 0.001  # Intra and inter class connection probabilities
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if y[i] == y[j]:
                if np.random.rand() < p_intra:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
            else:
                if np.random.rand() < p_inter:
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    return X, adjacency_matrix, y
num_classes=5
num_samples =5000
noise=.05
f = lambda x: 1
mode = "polar"
features, adjacency_matrix, labels = generate_graph(num_samples,num_classes, noise, f, mode)
features_tensor = torch.tensor(features, dtype=torch.float)
labels_tensor = torch.tensor(labels, dtype=torch.long)
edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)

colors = ['red','blue','green','yellow','purple']
def visualize_graph(features, adjacency_matrix, labels):
    G = nx.from_numpy_array(adjacency_matrix)
    pos = {i: (features[i, 0], features[i, 1]) for i in range(len(features))}
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label
    plt.figure(figsize=(10, 10))
    color_map = [colors[label%len(colors)] for label in labels]
    nx.draw_networkx_nodes(G, pos, node_color=color_map, alpha=0.6, edgecolors='w')
    
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # plt.title('')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()

visualize_graph(features, adjacency_matrix, labels)


# def visualize_graph(features, adjacency_matrix, labels):
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w')
    
#     num_nodes = len(features)
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if adjacency_matrix[i, j] == 1:
#                 plt.plot([features[i, 0], features[j, 0]], [features[i, 1], features[j, 1]], 'silver', lw=0.3, alpha=0.5)
    
#     plt.title('Graph')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     # plt.colorbar(scatter, label='Class')
#     plt.show()
# visualize_graph(features, adjacency_matrix, labels)



graph_data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor)


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


model = GCN(num_features=2, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.NLLLoss()

data_loader = DataLoader([graph_data], batch_size=32, shuffle=True)

def train():
    model.train()
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out,data.y)
        loss.backward()
        optimizer.step()


train()
num_epochs = 10000
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
    if (epoch+1)%50==0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')

def test(model,data):
    model.eval()
    with torch.no_grad():  
        out = model(data)
        print(out.shape)
        _, pred = out.max(dim=1)  
        correct = pred.eq(data.y).sum().item()  
        acc = correct / data.num_nodes  
    return acc

# After training, evaluate the model on the entire graph data
accuracy = test(model, graph_data)
print(f'Node-wise Accuracy: {accuracy:.4f}')