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
import warnings
import os,sys
import glob
from config import *
import gnn as g
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    ntrain = 1 # number of training graphs

    method = 'GD' # GD, SGD or Adam
    train = True # True to train the model. 
    sharpness = 20
    noiselevel = 0.2
    nepoch_pre = 50000 # Pre train: softmax, no multiplier updating
    nepoch = 500000 # Softmax, multiplier updating

    lr_pre = 1e1 # value for GD. For Adam it's around 1e-4
    penalty_pre = 1e-7 # time constraint strength in pre-training

    lr = 1e1 # value for GD. For Adam it's around 1e-4
    penalty_ini = 1e-7 # Differetial Multiplier Damping, initial
    penalty_end = 1e-4 # Differetial Multiplier Damping, final

    # Load data
    datadir = 'graphs-%s/'%case
    graphnames = glob.glob(datadir+'*.pt')
    graphs=[]
    for i in range(ntrain):
        igraph=torch.load(graphnames[i])
        print(graphnames[i])
        graphs.append(igraph)
    dataset = g.Loader(graphs_list=graphs)
    dataloader = DataLoader(dataset,batch_size=1)
    train_be = []
    train_bs = []
    train_bt = []
    for i_batch,graph in enumerate(dataloader):
        # only correct for batchsize=1
        be = torch.zeros(graph.edge_attr.size(0),dtype=torch.long).cuda()
        bs = torch.zeros(graph.x_h.size(0),dtype=torch.long).cuda()
        bt = torch.zeros(graph.x_g.size(0),dtype=torch.long).cuda()
        train_be.append(be)
        train_bs.append(bs)
        train_bt.append(bt)

    print('Start Pre-Training')
    tclass = torch.arange(2)*Tmax
    tclass = tclass.float().cuda()
    softmax = nn.Softmax(dim=-1)
    for i_batch,graph in enumerate(dataloader):
        if nepoch_pre>0:
            ### Pre Train ###
            gumbel = False
            # Load Model
            g_index = graph.edge_index[1]
            initial = -torch.log(Tmax/graph.x_g[g_index,0]/0.95-1).cpu()
 
            try:
                c_all = torch.load('brute_result/parms-%d.pth'%i_batch)
            except:
                print('No parameters saved')
                c_all = torch.nn.Parameter(initial.float().cuda(),requires_grad=True)
            if method == 'Adam':
                optimizer = optim.Adam([c_all],lr=lr_pre)
            if method == 'SGD':
                optimizer = optim.SGD([c_all],lr=lr_pre)
            for i_epoch in range(nepoch_pre):
                time = Tmax*torch.sigmoid(c_all)
                if train:
                    noise = noiselevel*(torch.rand(time.shape).float().cuda()-0.5)
                    time = time+noise
                    inttime = torch.floor(time)
                    time = inttime + torch.sigmoid(sharpness*(time-0.5-inttime))
                else:
                    time = torch.round(time)
                loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty_pre)
                loss.backward()

                if method == 'GD':
                    with torch.no_grad():
                        c_all -=  lr_pre*c_all.grad #Gradient Descent
                    if c_all.grad is not None:
                        c_all.grad.zero_()
                else:
                    optimizer.step()
                if i_epoch%100==0:
                    print("%d %.3f %.3f %.1f %.1f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))
            print('Pre-Training Finished')

        if nepoch>0:
            ### Softmax ###
            print('Start Training')
            # Load Model
            penalty = penalty_ini
            rate = (penalty_end/penalty_ini)**(1./nepoch_softmax)
            if method == 'Adam':
                optimizer = optim.Adam([c_all],lr=lr_pre)
            if method == 'SGD':
                optimizer = optim.SGD([c_all],lr=lr_pre)
            
            for i_epoch in range(nepoch:
                time = Tmax*torch.sigmoid(c_all)
                if train:
                    noise = noiselevel*(torch.rand(time.shape).float().cuda()-0.5)
                    time = time+noise
                    inttime = torch.floor(time)
                    time = inttime + torch.sigmoid(sharpness*(time-0.5-inttime))
                else:
                    time = torch.round(time)
                loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty)
                loss.backward()
                if i_epoch%100==0:
                    print("%d %.3f %.3f %.1f %.1f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))
                if method == 'GD':
                    with torch.no_grad():
                        c_all -=  lr_pre*c_all.grad #Gradient Descent
                    if c_all.grad is not None:
                        c_all.grad.zero_()
                else:
                    optimizer.step()
                penalty *= rate
            print('Training Finished')
        print('Result')
        time = Tmax*torch.sigmoid(c_all)
        time = torch.round(time)
        loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty,finaloutput=True)
        print("%d %.3f %.3f %.1f %.1f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))
        print(graphnames[i_batch])
        
        torch.save(c_all,'brute_result/parms-%d.pth'%i_batch)

