import torch
import numpy as np
import gnn

def to_Graph(indices,properties,maxtime=30):
    properties = np.array(properties)
    edge_attr = []
    e_s = [] # start from fibers
    e_t = [] # end at galaxies
    for i,index in enumerate(indices):
        for j in range(len(index)):
            if index[j]<2394: # galaxy i can be viewed by fiber index[j]
                e_s.append(index[j])
                e_t.append(i)
                edge_attr.append(np.zeros(gnn.F_e))
    edge_attr = torch.tensor(edge_attr).float()
    edge_index = torch.tensor([e_s,e_t],dtype=torch.long)
    x_s = torch.zeros(2394,gnn.F_xs).float()  #default value for vertices parameters, containing total time
    x_t = torch.tensor(properties).float()  #default value for vertices parameters, containing total time
    u=torch.tensor([np.zeros(gnn.F_u)]).float() #default value for global parameters
    data = gnn.BipartiteData(edge_index,x_s,x_t,edge_attr,u)
    return data

if __name__ == '__main__':
    ngraph = 25
    utils = np.loadtxt('utils.txt')
    for igraph in range(ngraph):
        indices = np.loadtxt('pairs/pair-%d.txt'%igraph,dtype=int)
        args = utils[:,0]>-1e9 # All True
        for i in range(len(indices)):
            index = indices[i]
            num = 0
            for j in range(len(index)):
                if index[j]<2394:
                    num+=1
            if num==0: #No fiber
                args[i] = False
        graph = to_Graph(indices[args],utils[args])
        torch.save(graph,'graphs/graph-%d.pt'%igraph)
