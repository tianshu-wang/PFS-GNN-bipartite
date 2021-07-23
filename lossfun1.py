import torch
import torch.nn as nn
from torch_scatter import scatter_mean,scatter
import torch.nn.functional as F

def Lossfun(time,graph,penalty=0,finaloutput=False):
    total_time = 42 # total number of visits
    # Hyperparameter
    leaky = nn.LeakyReLU(0.) # parameter here tunes the panelty difference between overtime and undertime
    # Batch
    bt = graph.batch
    src,tgt = graph.edge_index
    x_g = graph.x_g
    x_h = graph.x_h
    timereq = x_g[:,0]
    tgtclass = x_g[:,1:-1]
    totclass = torch.sum(tgtclass,dim=0)
    # Time per galaxy
    time_sum = scatter(time,tgt,dim_size=x_g.size(0),reduce='sum')
    # Time per fiber
    spent_time = scatter(time,src,dim_size=x_h.size(0),reduce='sum')
    # Overtime
    overtime = spent_time-total_time
    delta = leaky(overtime).view(1,2394) 
    # Penalty
    time_constraint = torch.sum(penalty*delta*delta)
    # Reward
    sharpness = 5
    observed = torch.sigmoid((time_sum-timereq+0.5)*sharpness)
    reward = (observed*tgtclass.T).T
    completeness = torch.sum(reward,dim=0)/torch.sum(tgtclass,dim=0)
    totnum = torch.sum(observed)
    totutils = torch.min(completeness)
    
    if finaloutput:
        sharpness = 1000
        observed = torch.sigmoid((time_sum-timereq+0.5)*sharpness)
        reward = (observed*tgtclass.T).T
        clsnum = torch.sum(reward,dim=0)
        clsall = torch.sum(tgtclass,dim=0)
        completeness = clsnum/clsall
        totnum = torch.sum(observed)
        totutils = torch.min(completeness)
        #print(clsnum) #number observed in each class
        #print(clsall) #total number in each class
    return -totutils+time_constraint,totutils,totnum,torch.sum(F.relu(overtime))/(2394*total_time),torch.sum(F.relu(-overtime))/(2394*total_time)
