import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data,DataLoader
from torch_scatter import scatter_mean,scatter
from torch.utils.data import Dataset
from torch.autograd import Variable
import gnn as g
import warnings
import os,sys
warnings.filterwarnings("ignore")

total_time = 6 # total number of visits


def Loss(time,graph,penalty=0,batchsize=1):
    # Hyperparameter
    leaky = nn.LeakyReLU(1.) # parameter here tunes the panelty difference between overtime and undertime
    # Batch
    bt = graph.batch
    src,tgt = graph.edge_index
    x_t = graph.x_t
    x_s = graph.x_s
    reward_set = x_t
    # Time per galaxy
    time_sum = scatter(time,tgt,dim_size=x_t.size(0),reduce='sum')
    # Time per fiber
    spent_time = scatter(time,src,dim_size=x_s.size(0),reduce='sum')
    # Overtime
    overtime = spent_time-total_time
    delta = leaky(overtime).view(batchsize,2394) #[batch,fibers]
    # Penalty
    time_constraint = torch.sum(penalty*delta*delta)
    # Utility Parameters
    numreq = 5000 # at least 5000 galaxies for each graph, requiring 5000 in total 2394*6=14364 visits
    avgrwd = 2 # average reward of galaxies observed for number requirement, should finally goes to infinity
    # Calculate Utilities
    # Per galaxy utilities
    galutils = ((time_sum)*(reward_set[:,0]-0)
               +F.relu(time_sum-1.)*(reward_set[:,1]-2*reward_set[:,0]+0)
               +F.relu(time_sum-2.)*(reward_set[:,2]-2*reward_set[:,1]+reward_set[:,0])
               +F.relu(time_sum-3.)*(reward_set[:,3]-2*reward_set[:,2]+reward_set[:,1]) 
               -F.relu(time_sum-4.)*reward_set[:,3]) #Piecewise linear. Constant when t>4
    totgalutils = torch.sum(galutils)
    # Global utilities
    requirement = reward_set[:,4] # 1 if satisfy the requirement, otherwise 0
    num = scatter(torch.sigmoid(5*(time_sum-0.5))*requirement,bt,reduce='sum') # at least 1 visit
    #print(num.item())
    numutils = avgrwd*numreq*torch.sigmoid(0.01*(num-numreq)) 
    totnumutils = torch.sum(numutils)
    # Total utilities
    totutils = totgalutils+totnumutils
    return -totutils+time_constraint,totgalutils,totnumutils,torch.sum(F.relu(overtime)),torch.sum(F.relu(-overtime))


if __name__ == '__main__':
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    ID = str(idx)
    
    ntrain = 25 # number of training graphs
    ntest = 0 # number of testing graphs
    batchsize = 1 # Currently can only be trained per graph. Can't have batchsize>1

    sharpness = 20
    noiselevel = [0.2,0.3,0.2,0.3][idx]

    train = False # True to train the model. 
    nepoch_pre = 0#2000 # Pre-train
    nepoch = 1 # Gradually increase the strength of overtime punishment

    lr_pre = 5e-4 # GNN LR
    penalty_pre = 1e-1 # time constraint strength in pre-training

    lr = [1e-4,1e-4,1e-4,1e-4][idx] # GNN LR
    penalty_ini = 1e0 # initial
    penalty_end = 1e0


    batchsize = min(batchsize,ntrain)
    # Load data
    utils = np.loadtxt('utils.txt') # utilities info for all mock galaxies
    Ntarget = len(utils)
    graphs=[]
    for i in range(ntrain):
        igraph=torch.load('graphs/graph-%d.pt'%i)
        graphs.append(igraph)
    dataset = g.Loader(graphs_list=graphs)
    dataloader = DataLoader(dataset,batch_size=batchsize)
    train_be = []
    train_bs = []
    train_bt = []
    for i_batch,graph in enumerate(dataloader):
        # only correct for batchsize=1
        be = torch.zeros(graph.edge_attr.size(0),dtype=torch.long).cuda()
        bs = torch.zeros(graph.x_s.size(0),dtype=torch.long).cuda()
        bt = torch.zeros(graph.x_t.size(0),dtype=torch.long).cuda()
        train_be.append(be)
        train_bs.append(bs)
        train_bt.append(bt)
    test_graphs = []
    all_rewards = []
    all_args = []
    for igraph in range(ntest):
        i_test_graph = torch.load('graphs/graph-%d.pt'%(ntrain+igraph))
        test_graphs.append(i_test_graph)
        indices = np.loadtxt('pairs/pair-%d.txt'%(ntrain+igraph),dtype=int)
        args0 = utils[:,0]>-1e9 # All True
        args3 = utils[:,0]>-1e9 # All True
        for i in range(len(indices)):
            index = indices[i]
            num = 0
            for j in range(len(index)):
                if index[j]<2394:
                    num+=1
            if num==0: #No fiber
                args0[i] = False
            if num==3: #3 fiber
                args3[i] = False
        args = args0*args3
        reward_set = torch.from_numpy(utils[args]).float().cuda()
        all_rewards.append(reward_set)
        all_args.append(args)
        edge_attr = i_test_graph.edge_attr
    test_set = g.Loader(graphs_list=test_graphs)
    testloader = DataLoader(test_set,batch_size=1)
    test_be = []
    test_bs = []
    test_bt = []
    for i_batch,graph in enumerate(dataloader):
        # only correct for batchsize=1
        be = torch.zeros(graph.edge_attr.size(0),dtype=torch.long).cuda()
        bs = torch.zeros(graph.x_s.size(0),dtype=torch.long).cuda()
        bt = torch.zeros(graph.x_t.size(0),dtype=torch.long).cuda()
        test_be.append(be)
        test_bs.append(bs)
        test_bt.append(bt)
    
    print('Start Pre-Training')
    if nepoch_pre>0:
        ### Pre Train ###
        # Load Model
        gnn = g.GNN().cuda()
        gnn.sharpness = sharpness
        gnn.noiselevel = noiselevel
        print('Num of parameters:',sum(p.numel() for p in gnn.parameters() if p.requires_grad))
        optimizer = optim.Adam(gnn.parameters(),lr=lr_pre)
        try:
            gnn.load_state_dict(torch.load('model_gnn_pre'+ID+'.pth'))
            gnn.eval()
        except:
            print('No available GNN model exists')
        for i_epoch in range(nepoch_pre):
            for i_batch,graph in enumerate(dataloader):
                gnn.zero_grad()
                time,edge_index = gnn(graph,train_be[i_batch],train_bs[i_batch],train_bt[i_batch])
                loss,utils,numutils,overtime,undertime = Loss(time,graph,penalty=penalty_pre,batchsize=batchsize)
                loss.backward()
                print("%d %.1f %.1f %.1f -%.1f +%.1f"%(i_batch,-loss.item()/batchsize,utils.item()/batchsize,numutils.item()/batchsize,overtime.item()/batchsize,undertime.item()/batchsize))
                if train:
                    optimizer.step()
        torch.save(gnn.state_dict(), 'model_gnn_pre'+ID+'.pth')
        print('Pre-Training Finished')

    if nepoch>0:
        print('Start Training')
        # Load Model
        gnn = g.GNN().cuda()
        gnn.sharpness = sharpness
        gnn.noiselevel = noiselevel
        print('Num of parameters:',sum(p.numel() for p in gnn.parameters() if p.requires_grad))
        optimizer = optim.Adam(gnn.parameters(),lr=lr)
        
        try:
            gnn.load_state_dict(torch.load('model_gnn'+ID+'.pth'))
            gnn.eval()
        except:
            print('No available GNN model found. Use pre-trained model')
            gnn.load_state_dict(torch.load('model_gnn_pre'+ID+'.pth'))
            gnn.eval()
        penalty = penalty_ini
        rate = (penalty_end/penalty_ini)**(1./nepoch)
        
        for i_epoch in range(nepoch):
            for i_batch,graph in enumerate(dataloader):
                gnn.zero_grad()
                time,edge_index = gnn(graph,train_be[i_batch],train_bs[i_batch],train_bt[i_batch])
                loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty,batchsize=batchsize)
                #print(extra_loss.item())
                loss.backward()
                if train:
                    optimizer.step()
                print("%d %.1f %.1f %.1f -%.1f +%.1f"%(i_batch,-loss.item()/batchsize,utils.item()/batchsize,numutils.item()/batchsize,overtime.item()/batchsize,undertime.item()/batchsize))
            penalty *= rate
        print('Result')
        for i_batch,graph in enumerate(dataloader):
            time,edge_index = gnn(graph,train_be[i_batch],train_bs[i_batch],train_bt[i_batch],train=False)
            loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty,batchsize=batchsize)
            print("%d %.1f %.1f %.1f -%.1f +%.1f"%(i_batch,-loss.item()/batchsize,utils.item()/batchsize,numutils.item()/batchsize,overtime.item()/batchsize,undertime.item()/batchsize))

        torch.save(gnn.state_dict(), 'model_gnn'+ID+'.pth')
        print('Training Finished')

