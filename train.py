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
warnings.filterwarnings("ignore")

total_time = 6 # total number of visits


def Loss(time,graph,c_t,damp=0,batchsize=1):
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
    delta = leaky(spent_time-total_time).view(batchsize,2394) #[batch,fibers]
    # Penalty
    time_constraint = torch.sum((c_t+damp*delta.detach())*delta)
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
    numutils = avgrwd*numreq*torch.sigmoid(0.1*(num-numreq)) 
    totnumutils = torch.sum(numutils)
    # Total utilities
    totutils = totgalutils+totnumutils
    #print(torch.sum(reward_set[:,4]*reward_set[:,3]).item())
    return -totutils+time_constraint,totgalutils,totnumutils,torch.sum(delta)


if __name__ == '__main__':
    ntrain = 20 # number of training graphs
    ntest = 5 # number of testing graphs
    batchsize = 1 # Currently can only be trained per graph. Can't have batchsize>1

    train = True # True to train the model. 
    brute_force = True
    nepoch_pre = 1000 # Pre train: softmax, no multiplier updating
    nepoch_softmax = 0 # Softmax, multiplier updating
    nepoch_gumbel = 0 # Gumbel, multiplier updating

    lr_pre = 1e-5 # GNN LR
    damp_pre = 1e-2 # time constraint strength in pre-training

    lr_soft = 1e-6 # GNN LR
    lr_c_soft = 1e-5 # Multiplier LR
    damp_soft_ini = 1e-1 # Differetial Multiplier Damping, initial
    damp_soft_end = 1e1 # Differetial Multiplier Damping, final

    lr_gumbel = 1e-6 # GNN LR
    lr_c_gumbel = 1e-6 # Multiplier LR
    damp_gumbel_ini = 1e-1 # Differetial Multiplier Damping, initial
    damp_gumbel_end = 1e-0 # Differetial Multiplier Damping, final

    c_e_pre = 0 # extra loss term weight
    c_e_soft_ini = 1e-4 # extra loss term weight
    c_e_soft_end = 1e0 # extra loss term weight


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

    if train:
        softmax = nn.Softmax(dim=-1)
        print('Start Pre-Training')
        if nepoch_pre>0:
            ### Pre Train ###
            gumbel = False
            # Load Model
            gnn = g.GNN(gumbel=gumbel).cuda()
            print('Num of parameters:',sum(p.numel() for p in gnn.parameters() if p.requires_grad))
            optimizer = optim.Adam(gnn.parameters(),lr=lr_pre)
            try:
                gnn.load_state_dict(torch.load('model_gnn_pre.pth'))
                gnn.eval()
            except:
                print('No available GNN model exists')
            c_t = torch.nn.Parameter(0.0*torch.ones(2394).float().cuda(),requires_grad=True)
            for i_epoch in range(nepoch_pre):
                for i_batch,graph in enumerate(dataloader):
                    gnn.zero_grad()
                    time,edge_index,extra_loss,edge_attr = gnn(graph,train_be[i_batch],train_bs[i_batch],train_bt[i_batch])
                    loss,utils,numutils,overtime= Loss(time,graph,c_t,damp=damp_pre,batchsize=batchsize)
                    if brute_force:
                        c_all = torch.load('brute_result/parms-%d.pth'%i_batch)
                        loss_1 = torch.sum((softmax(edge_attr)-softmax(c_all))**2)
                        loss_1.backward()
                        print(loss_1.item())
                    else:
                        loss += c_e_pre*extra_loss
                        loss.backward()
                    print("%d %.1f %.1f %.1f %.1f"%(i_batch,-loss.item()/batchsize,utils.item()/batchsize,numutils.item()/batchsize,overtime.item()/batchsize))
                    optimizer.step()
                    #print(loss_1.item())
                if c_t.grad is not None:
                    c_t.grad.zero_()
            torch.save(gnn.state_dict(), 'model_gnn_pre.pth')
            print('Pre-Training Finished')

        if nepoch_softmax>0:
            ### Softmax ###
            print('Start Softmax Training')
            gumbel = False
            # Load Model
            gnn = g.GNN(gumbel=gumbel).cuda()
            print('Num of parameters:',sum(p.numel() for p in gnn.parameters() if p.requires_grad))
            optimizer = optim.Adam(gnn.parameters(),lr=lr_soft)
            
            try:
                gnn.load_state_dict(torch.load('model_gnn_soft.pth'))
                gnn.eval()
            except:
                print('No available GNN model exists')
                gnn.load_state_dict(torch.load('model_gnn_pre.pth'))
                gnn.eval()
            c_t = torch.nn.Parameter(0.0*torch.ones(2394).float().cuda(),requires_grad=True)
            damp_soft = damp_soft_ini
            rate_soft = (damp_soft_end/damp_soft_ini)**(1./nepoch_softmax)
            c_e_soft = c_e_soft_ini
            c_e_rate_soft = (c_e_soft_end/c_e_soft_ini)**(1./nepoch_softmax)
            
            for i_epoch in range(nepoch_softmax):
                for i_batch,graph in enumerate(dataloader):
                    gnn.zero_grad()
                    time,edge_index,extra_loss,edge_attr = gnn(graph,train_be[i_batch],train_bs[i_batch],train_bt[i_batch])
                    loss,utils,numutils,overtime= Loss(time,graph,c_t,damp=damp_soft,batchsize=batchsize)
                    loss += c_e_soft*extra_loss
                    print(extra_loss.item())
                    loss.backward()
                    optimizer.step()
                    print("%d %.1f %.1f %.1f %.1f"%(i_batch,-loss.item()/batchsize,utils.item()/batchsize,numutils.item()/batchsize,overtime.item()/batchsize))
                with torch.no_grad():
                    c_t +=  lr_c_soft*c_t.grad
                    c_t = F.relu(c_t).requires_grad_(True)
                if c_t.grad is not None:
                    c_t.grad.zero_()
                damp_soft *= rate_soft
                c_e_soft *= c_e_rate_soft
    
            torch.save(gnn.state_dict(), 'model_gnn_soft.pth')
            print('Softmax Training Finished')

        if nepoch_gumbel>0:
            ### Gumbel ###        
            print('Start Gumbel Training')
            gumbel = True
            # Load Model
            gnn = g.GNN(gumbel=gumbel).cuda()
            print('Num of parameters:',sum(p.numel() for p in gnn.parameters() if p.requires_grad))
            optimizer = optim.Adam(gnn.parameters(),lr=lr_gumbel)
            
            try:
                gnn.load_state_dict(torch.load('model_gnn.pth'))
                gnn.eval()
            except:
                print('No available GNN model exists. Use softmax model')
                gnn.load_state_dict(torch.load('model_gnn_soft.pth'))
                gnn.eval()
            c_t = torch.nn.Parameter(0.0*torch.ones(2394).float().cuda(),requires_grad=True)
            damp_gumbel = damp_gumbel_ini
            rate_gumbel = (damp_gumbel_end/damp_gumbel_ini)**(1./nepoch_gumbel)
            for i_epoch in range(nepoch_gumbel):
                for i_batch,graph in enumerate(dataloader):
                    gnn.zero_grad()
                    time,edge_index,extra_loss,edge_attr = gnn(graph,train_be[i_batch],train_bs[i_batch],train_bt[i_batch])
                    loss,utils,numutils,overtime= Loss(time,graph,c_t,damp=damp_gumbel,batchsize=batchsize)
                    loss.backward()
                    optimizer.step()
                    print("%d %.1f %.1f %.1f %.1f"%(i_batch,-loss.item()/batchsize,utils.item()/batchsize,numutils.item()/batchsize,overtime.item()/batchsize))
                with torch.no_grad():
                    c_t +=  lr_c_gumbel*c_t.grad
                    c_t = F.relu(c_t).requires_grad_(True)
                if c_t.grad is not None:
                    c_t.grad.zero_()
                damp_gumbel *= rate_gumbel
            torch.save(gnn.state_dict(), 'model_gnn.pth')
            print('Gumbel Training Finished')
