# Non-Linear-Graph-Neural-Network-Class-Identification


Generating synthetic graphs and then training GNNS on class identification for a given node in the graph. 

Scripts in src/:

- boringgraph.py creates linearly separable graph data

- data_gen.py creates nonlinear graph data and trains a gnn on class identification for single uses and writes the data into test_runs/

- dataTEST.py similar to data_gen, but with a few more features for analysis on individual GNN's

- heatmap.py takes data from test_runs/ and generates a heatmap of GNN performance (TODO) Heatmaps are then saved in Images/

- pandas_data.py simply plots csv data on a plot without the heatmap


