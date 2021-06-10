# PFS-GNN-bipartite
to_graph: Convert the connectivity information (pair.txt) and the galaxy information (utils.txt) into graphs (graph.pt)

gnn.py: Define the bipartite graph class and GNN strutures

train.py: Train the GNN with an arbitrary reward (loss) function.

brute.py: Brute force method

# Example of input data 
pair.txt: Relation between targets and fibers. Each line corresponds to a galaxy in the field. The three numbers in the line are the fiber indices that this galaxy can be visited. Valid fiber indices are 0-2393, and 2394 means N/A.

graph.pt: The graph generated from pair.txt by to_graph.py

There should be another file called utils.txt which contains all galaxy information required to calculate the final reward function, but it is too large to be uploaded here. It has the same number of lines as the pair.txt, i.e., the ith line in pair.txt is correpsonding to the ith line in utils.txt.
