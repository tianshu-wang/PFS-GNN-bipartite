# PFS-GNN-bipartite

The data are too large for github. Only two example graphs (one for each case) are uploaded here.

To run the code, choose one of the config file, then run "python train.py". The graphs are constructed for gpu so you need a gpu to use them.

# Brief Descriptions of the Files

gnn.py : Define the bipartite graph and the GNN.

train.py : The main file to run the code.

/config : Contain the config files for both cases. To use them, copy the config file to the same directory as the train.py and rename it as config.py

brute.py : GD method.

construct_graph.py : Construct the graph. The connectivity and the initialization of the graph will depend on the problem.

load.py: Load the graphs. Currently we only support batchsize = 1. To do a batched training, merge multiple graphs into a single one in the graph construction step.


