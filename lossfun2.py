import torch
import torch.nn as nn
from torch_scatter import scatter_mean,scatter
import torch.nn.functional as F

def Lossfun(time,graph,penalty=0,finaloutput=False):
    total_time = 6 # total number of visits
    tmax = 4 # No extra gain if a target receives more than 4 visits. 
             # Different from the meaning of Tmax in config.py, though they are equal in this case.
    # Hyperparameter
    leaky = nn.LeakyReLU(1.) # parameter here tunes the panelty difference between overtime and unused time
    # Batch
    bt = graph.batch
    src,tgt = graph.edge_index
    x_g = graph.x_g
    x_h = graph.x_h
    reward_set = x_g
    # Time per galaxy
    time_sum = scatter(time,tgt,dim_size=x_g.size(0),reduce='sum')  
    time_sum = time_sum - F.relu(time_sum-tmax)
    # Time per fiber
    spent_time = scatter(time,src,dim_size=x_h.size(0),reduce='sum')
    # Overtime
    overtime = spent_time-total_time
    delta = leaky(overtime).view(1,2394) #[batch,fibers]
    # Penalty
    time_constraint = torch.sum(penalty*delta*delta)
    # Utility Parameters
    numreq = 5000 # at least 5000 galaxies for each graph, requiring 5000 in total 2394*6=14364 visits
    avgrwd = 2 # average reward of galaxies observed for number requirement, should finally goes to infinity
    # Per galaxy utilities
    galutils = ((time_sum)*(reward_set[:,0]-0)
               +F.relu(time_sum-1.)*(reward_set[:,1]-2*reward_set[:,0]+0)
               +F.relu(time_sum-2.)*(reward_set[:,2]-2*reward_set[:,1]+reward_set[:,0])
               +F.relu(time_sum-3.)*(reward_set[:,3]-2*reward_set[:,2]+reward_set[:,1]) 
               ) #Piecewise linear. Constant when t>4
    totgalutils = torch.sum(galutils)
    # Global utilities
    requirement = reward_set[:,4] # 1 if satisfy the requirement, otherwise 0
    num = scatter(torch.sigmoid(5*(time_sum-0.5))*requirement,bt,reduce='sum') # at least 1 visit
    numutils = avgrwd*numreq*torch.sigmoid(0.01*(num-numreq)) 
    totnumutils = torch.sum(numutils)
    # Total utilities
    totutils = totgalutils+totnumutils
    return -totutils+time_constraint,totgalutils,totnumutils,torch.sum(F.relu(overtime))/(2394*total_time),torch.sum(F.relu(-overtime))/(2394*total_time)
