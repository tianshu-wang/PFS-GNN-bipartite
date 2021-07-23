# PFS-GNN-bipartite

gnn.py : Define the bipartite graph and the GNN.

train.py : The main file to run the code.

/config : Contain the config files for both cases. To use them, copy the config file to the same directory as the train.py and rename it as config.py

brute.py : Brute force method.

to_graph : Construct the graph. The connectivity and the initialization of the graph will depend on the problem.

load.py: Load the graphs. Currently we only support batchsize = 1. To do a batched training, simply concatenate multiple graphs in the graph construction step.

The data are too large for github, so they are not included here.
