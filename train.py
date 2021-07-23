import numpy as np
import torch
from torch import optim
from torch_scatter import scatter_mean,scatter
import gnn as g
import warnings
import os,sys
from config import *
from load import *
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    print("Column Names:")
    print("GraphID, Total Loss, Reward, Overtime, Undertime")
    if train:
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
                gnn.load_state_dict(torch.load(modeldir+pre_modelname))
                gnn.eval()
            except:
                print('No available GNN model exists')
            for i_epoch in range(nepoch_pre):
                for i_batch,graph in enumerate(dataloader):
                    gnn.zero_grad()
                    time,edge_index = gnn(graph,train_be[i_batch],train_bh[i_batch],train_bg[i_batch])
                    loss,utils,numutils,overtime,undertime = Loss(time,graph,penalty=penalty_pre)
                    loss.backward()
                    if train:
                        optimizer.step()
                    if i_epoch%100==0:
                        print("%d %.3f %.3f +%.3f -%.3f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))
            torch.save(gnn.state_dict(), modeldir+pre_modelname)
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
                gnn.load_state_dict(torch.load(modeldir+modelname))
                gnn.eval()
            except:
                print('No available GNN model found. Use pre-trained model')
                gnn.load_state_dict(torch.load(modeldir+pre_modelname))
                gnn.eval()
            penalty = penalty_ini
            rate = (penalty_end/penalty_ini)**(1./nepoch)
            
            for i_epoch in range(nepoch):
                for i_batch,graph in enumerate(dataloader):
                    gnn.zero_grad()
                    time,edge_index = gnn(graph,train_be[i_batch],train_bh[i_batch],train_bg[i_batch])
                    loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty)
                    #print(extra_loss.item())
                    loss.backward()
                    if train:
                        optimizer.step()
                    if i_epoch%100==0:
                        print("%d %.3f %.3f +%.3f -%.3f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))
                penalty *= rate
            print('Result')
            for i_batch,graph in enumerate(dataloader):
                time,edge_index = gnn(graph,train_be[i_batch],train_bh[i_batch],train_bg[i_batch],train=False)
                loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty,finaloutput=True)
                print("%d %.3f %.3f +%.3f -%.3f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))
            print('Validation')
            for i_batch,graph in enumerate(validloader):
                time,edge_index = gnn(graph,valid_be[i_batch],valid_bh[i_batch],valid_bg[i_batch],train=False)
                loss,utils,numutils,overtime,undertime= Loss(time,graph,penalty=penalty,finaloutput=True)
                print("%d %.3f %.3f +%.3f -%.3f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))

            torch.save(gnn.state_dict(), modeldir+modelname)
            print('Training Finished')
    else:
        gnn = g.GNN().cuda()
        try:
            gnn.load_state_dict(torch.load(modeldir+modelname))
            gnn.eval()
        except:
            try:
                gnn.load_state_dict(torch.load(modeldir+pre_modelname))
                gnn.eval()
                print('No available GNN model found. Use pre-trained model')
            except:
                print('No saved model found. Use random model')
        print('Num of parameters:',sum(p.numel() for p in gnn.parameters() if p.requires_grad))
        for i_batch,graph in enumerate(dataloader):
            time,edge_index = gnn(graph,train_be[i_batch],train_bh[i_batch],train_bg[i_batch],train=False)
            loss,utils,numutils,overtime,undertime= Loss(time,graph,finaloutput=True)
            print("%d %.3f %.3f +%.3f -%.3f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))
        print('Validation')
        for i_batch,graph in enumerate(validloader):
            time,edge_index = gnn(graph,valid_be[i_batch],valid_bh[i_batch],valid_bg[i_batch],train=False)
            loss,utils,numutils,overtime,undertime= Loss(time,graph,finaloutput=True)
            print("%d %.3f %.3f +%.3f -%.3f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))

        print('Test')
        for i_batch,graph in enumerate(testloader):
            time,edge_index = gnn(graph,test_be[i_batch],test_bh[i_batch],test_bg[i_batch],train=False)
            loss,utils,numutils,overtime,undertime= Loss(time,graph,finaloutput=True)
            print("%d %.3f %.3f +%.3f -%.3f"%(i_batch,-loss.item(),utils.item(),overtime.item(),undertime.item()))


