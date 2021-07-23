import torch
import numpy as np
import gnn
import glob
from config import *
import os

def to_Graph(indices,properties,maxtime=30):
    properties = np.array(properties)
    edge_attr = []
    e_h = [] # start from fibers
    e_g = [] # end at galaxies
    k=0
    reachable = np.zeros(len(properties),dtype=bool)
    for i,index in enumerate(indices):
        flag = False
        for j in range(len(index)):
            if index[j]<2394: # galaxy i can be viewed by fiber index[j]
                flag = True
                e_h.append(index[j])
                e_g.append(k)
                edge_attr.append(np.zeros(F_e))
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
    edge_index = torch.tensor([e_h,e_g],dtype=torch.long)
    #args = torch.argsort(edge_index[0]) 
    #edge_attr = edge_attr[args]
    #edge_index = edge_index[:,args]

    x_h = torch.zeros(2394,F_xh).float()
    x_g = torch.tensor(properties[reachable]).float()
    u=torch.tensor([np.zeros(F_u)]).float()
    data = gnn.BipartiteData(edge_index.cuda(),x_h.cuda(),x_g.cuda(),edge_attr.cuda(),u.cuda())
    return data

if __name__ == '__main__':
    names = glob.glob('pairs-%s/pair-*'%case)
    utils = np.loadtxt('initial_info/utils-%s.txt'%case)

    if not os.path.exists('graphs-%s/'%case):
        os.system('mkdir graphs-%s'%case)
    
    for k in range(len(names)):
        indices = np.loadtxt(names[k],dtype=int)
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
        torch.save(graph,names[k].replace('pairs-%s/pair'%case,'graphs-%s/graph'%case).replace('txt','pt'))
