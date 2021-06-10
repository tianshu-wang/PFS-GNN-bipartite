import torch
import numpy as np
import gnn

def to_Graph(indices,properties,maxtime=30):
    properties = np.array(properties)
    edge_attr = []
    e_s = [] # start from fibers
    e_t = [] # end at galaxies
    k=0
    reachable = np.zeros(len(properties),dtype=bool)
    for i,index in enumerate(indices):
        flag = False
        for j in range(len(index)):
            if index[j]<2394: # galaxy i can be viewed by fiber index[j]
                flag = True
                e_s.append(index[j])
                e_t.append(k)
                edge_attr.append(np.zeros(gnn.F_e))
        if flag:
            reachable[i]=True
            k+=1
    '''
    edge_index = torch.tensor([e_s,e_t],dtype=torch.long)
    esec = []
    temp = torch.tensor(e_s,dtype=torch.long)
    for i_src_id in range(max(e_s)+1):
        esec.append(len(temp[temp==i_src_id]))
    n_regular = max(esec)
    print(n_regular)
    flag = False
    for i_src_id,i_esec in enumerate(esec):
        if i_esec<n_regular:
            flag = True 
            e_s.extend([i_src_id]*(n_regular-i_esec))
            e_t.extend([len(indices)]*(n_regular-i_esec))
            edge_attr.extend(np.zeros(((n_regular-i_esec),gnn.F_e)))
    if flag:
        properties = np.append(properties,[np.zeros_like(properties[0])],axis=0)
    '''
    edge_attr = torch.tensor(edge_attr).float()
    edge_index = torch.tensor([e_s,e_t],dtype=torch.long)
    args = torch.argsort(edge_index[0]) # sort the edges by fiber id
    edge_attr = edge_attr[args]
    edge_index = edge_index[:,args]

    x_s = torch.zeros(2394,gnn.F_xs).float()
    x_t = torch.tensor(properties[reachable]).float()
    u=torch.tensor([np.zeros(gnn.F_u)]).float()
    data = gnn.BipartiteData(edge_index.cuda(),x_s.cuda(),x_t.cuda(),edge_attr.cuda(),u.cuda())
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
