import torch
import numpy as np
import gnn
import glob
from config import *
import os

# This function varies from problem to problem. 
# The following example only works for the PFS problem
def to_Graph(indices,properties):
    # properties: galaxy properties useful for g nodes
    # indices: pre-calculated connectivity
    properties = np.array(properties)
    edge_attr = []
    e_h = [] # start from h nodes
    e_g = [] # end at g nodes

    # Graph Connectivity Related to the Problem
    for i,index in enumerate(indices):
        for j in range(len(index)):
            if index[j]<2394: 
                e_h.append(index[j])
                e_g.append(k)
                edge_attr.append(np.zeros(F_e)) # Edge initialization

    edge_attr = torch.tensor(edge_attr).float()
    edge_index = torch.tensor([e_h,e_g],dtype=torch.long)

    x_h = torch.zeros(2394,F_xh).float()
    x_g = torch.tensor(properties[reachable]).float()
    u=torch.tensor([np.zeros(F_u)]).float()
    data = gnn.BipartiteData(edge_index.cuda(),x_h.cuda(),x_g.cuda(),edge_attr.cuda(),u.cuda())
    return data

if __name__ == '__main__':
    names = glob.glob('pairs-%s/pair-*'%case) # pre-calculated connectivity
    utils = np.loadtxt('initial_info/utils-%s.txt'%case) # pre-calculated galaxy properties

    if not os.path.exists('graphs-%s/'%case):
        os.system('mkdir graphs-%s'%case)
                   
    graph = to_Graph(indices[args],utils[args])
    torch.save(graph,names[k].replace('pairs-%s/pair'%case,'graphs-%s/graph'%case).replace('txt','pt'))
