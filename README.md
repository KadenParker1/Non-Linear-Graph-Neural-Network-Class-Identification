# Non-Linear-Graph-Neural-Network-Class-Identification


Generating synthetic graphs and then training GNNS on class identification for a given node in the graph. 

Scripts in src/:

- boringgraph.py creates linearly separable graph data

- graph_gen.py contains functions to generate graph and feature data in 2 or more classes

- gnn.py contains GNN architectures and training functions

- experiment.py contains functions to test GNN performance with various architectures and graph structures

- heatmap.py takes data from test_runs/ and generates a heatmap of GNN performance (TODO) Heatmaps are then saved in Images/

- pandas_data.py simply plots csv data on a plot without the heatmap


