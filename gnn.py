import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean,scatter
from torch_geometric.data import Data,Dataset,DataLoader
from torch_geometric.nn import MessagePassing
from typing import List
import warnings
import numpy as np
from config import *
warnings.filterwarnings("ignore")

n_x_out = 1
class BipartiteData(Data):
    def __init__(self, edge_index, x_h, x_g, edge_attr,u):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index.cuda()
        self.x_h = x_h.cuda()
        self.x_g = x_g.cuda()
        self.edge_attr = edge_attr.cuda()
        self.u = u.cuda()
        self.num_nodes = len(self.x_g) # useless, just to avoid warnings
    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.x_h.size(0)], [self.x_g.size(0)]]).cuda()
        else:
            return super().__inc__(key, value)

class Loader(Dataset):
    def __init__(self,graphs_list=None):
        self.graphs_list = graphs_list
    def __len__(self):
        return len(self.graphs_list)
    def __getitem__(self,idx):
        graph = self.graphs_list[idx]
        return graph

class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(n_h+n_g+n_x+n_u,n_x), LeakyReLU(0.1), Lin(n_x,n_x))
    def forward(self, x_h, x_g, edge_index, edge_attr, u, batch_e):
        src, tgt = edge_index
        out = torch.cat([x_h[src], x_g[tgt], edge_attr, u[batch_e]], 1)
        out =  self.edge_mlp(out)
        return out
class EdgeModel_out(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_out, self).__init__()
        self.edge_mlp = Seq(Lin(n_h+n_g+n_x+n_u,n_x_out), LeakyReLU(0.1), Lin(n_x_out,n_x_out))
    def forward(self, x_h, x_g, edge_index, edge_attr, u, batch_e):
        src, tgt = edge_index
        out = torch.cat([x_h[src], x_g[tgt], edge_attr, u[batch_e]], 1)
        out =  self.edge_mlp(out)
        return out
class HModel(torch.nn.Module):
    def __init__(self):
        super(HModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(n_x+n_g,n_x+n_g), LeakyReLU(0.1), Lin(n_x+n_g,n_x+n_g))
        self.node_mlp_2 = Seq(Lin(n_h+4*(n_x+n_g)+1+n_u,n_h), LeakyReLU(0.1), Lin(n_h,n_h))
    def forward(self, x_h, x_g, edge_index, edge_attr, u, batch_h):
        src, tgt = edge_index
        out = edge_attr
        out = torch.cat([x_g[tgt],edge_attr], dim=1)
        out = self.node_mlp_1(out)
        ns = torch.ones(len(out),1).float().cuda() 
        n = scatter(ns, src, dim=0, dim_size=x_h.size(0),reduce='sum')
        #a = scatter(out, src, dim=0, dim_size=x_h.size(0),reduce='sum') # mu
        a = scatter(out, src, dim=0, dim_size=x_h.size(0),reduce='mean') # mu
        b = torch.sqrt(1e-6+F.relu(scatter(out**2, src, dim=0, dim_size=x_h.size(0),reduce='mean')-a**2)) # sigma
        c = scatter((out-a[src])**3, src, dim=0, dim_size=x_h.size(0),reduce='mean')/b**3 #skewness
        d = scatter((out-a[src])**4, src, dim=0, dim_size=x_h.size(0),reduce='mean')/b**4 #kurtosis
        #out = torch.cat([x_h,a,u[batch_h]], dim=1)
        out = torch.cat([x_h,n,a,b,c,d,u[batch_h]], dim=1)
        out = self.node_mlp_2(out)
        return out
class GModel(torch.nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(n_x+n_h,n_x+n_h), LeakyReLU(0.1), Lin(n_x+n_h,n_x+n_h))
        self.node_mlp_2 = Seq(Lin(n_g+1*(n_x+n_h)+n_u,n_g), LeakyReLU(0.1), Lin(n_g,n_g))
    def forward(self, x_h, x_g, edge_index, edge_attr, u, batch_g):
        src, tgt = edge_index
        out = edge_attr
        out = torch.cat([x_h[src],edge_attr], dim=1)
        out = self.node_mlp_1(out)
        ns = torch.ones(len(out),1).float().cuda() 
        a = scatter(out, tgt, dim=0, dim_size=x_g.size(0),reduce='sum') # mu
        out = torch.cat([x_g,a,u[batch_g]], dim=1)
        out = self.node_mlp_2(out)
        return out
class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(n_u+n_h+n_g,n_u), LeakyReLU(0.1), Lin(n_u, n_u))
    def forward(self, x_h, x_g, edge_index, edge_attr, u, batch_h,batch_g):
        out = torch.cat([u, scatter_mean(x_h,batch_h,dim=0),scatter_mean(x_g,batch_g,dim=0)], dim=1)
        return self.global_mlp(out)
class Block(torch.nn.Module):
    def __init__(self,edge_model=None,h_model=None,g_model=None,u_model=None):
        super(Block, self).__init__()
        self.edge_model = edge_model
        self.h_model = h_model 
        self.g_model = g_model
        self.global_model = u_model
    def forward(self,x_h,x_g,edge_index,edge_attr,u,batch_e,batch_h,batch_g):
        if self.edge_model is not None:
            edge_attr = self.edge_model(x_h, x_g, edge_index, edge_attr, u, batch_e)
        if self.h_model is not None:
            x_h = self.h_model(x_h, x_g, edge_index, edge_attr, u, batch_h)
        if self.g_model is not None:
            x_g = self.g_model(x_h, x_g, edge_index, edge_attr, u, batch_g)
        if self.global_model is not None:
            u = self.global_model(x_h, x_g, edge_index, edge_attr, u, batch_h,batch_g)
        return x_h,x_g,edge_attr,u
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN,self).__init__()
        self.sharpness = 20
        self.noiselevel = 0.3
        self.classes = torch.arange(n_x_out).float().cuda()
        self.block_1 = Block(EdgeModel(),HModel(),GModel(),GlobalModel())
        self.bn_xh_1 = nn.BatchNorm1d(n_h)
        self.bn_xg_1 = nn.BatchNorm1d(n_g)
        self.bn_e_1 = nn.BatchNorm1d(n_x)

        self.block_2 = Block(EdgeModel(),HModel(),GModel(),GlobalModel())
        self.bn_xh_2 = nn.BatchNorm1d(n_h)
        self.bn_xg_2 = nn.BatchNorm1d(n_g)
        self.bn_e_2 = nn.BatchNorm1d(n_x)

        self.block_3 = Block(EdgeModel(),HModel(),GModel(),GlobalModel())
        self.bn_xh_3 = nn.BatchNorm1d(n_h)
        self.bn_xg_3 = nn.BatchNorm1d(n_g)
        self.bn_e_3= nn.BatchNorm1d(n_x)

        self.block_4 = Block(EdgeModel(),HModel(),GModel(),GlobalModel())
        self.bn_xh_4 = nn.BatchNorm1d(n_h)
        self.bn_xg_4 = nn.BatchNorm1d(n_g)
        self.bn_e_4= nn.BatchNorm1d(n_x)

        self.block_last = Block(EdgeModel_out())
    def forward(self,data,batch_e,batch_h,batch_g,train=True):
        x_h = data.x_h
        x_g = data.x_g
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u


        x_h,x_g,edge_attr,u = self.block_1(x_h,x_g,edge_index,edge_attr,u,batch_e,batch_h,batch_g)
        x_h = self.bn_xh_1(x_h)
        x_g = self.bn_xg_1(x_g)
        edge_attr = self.bn_e_1(edge_attr)

        x_h,x_g,edge_attr,u = self.block_2(x_h,x_g,edge_index,edge_attr,u,batch_e,batch_h,batch_g)
        x_h = self.bn_xh_2(x_h)
        x_g = self.bn_xg_2(x_g)
        edge_attr = self.bn_e_2(edge_attr)

        x_h,x_g,edge_attr,u = self.block_3(x_h,x_g,edge_index,edge_attr,u,batch_e,batch_h,batch_g)
        x_h = self.bn_xh_3(x_h)
        x_g = self.bn_xg_3(x_g)
        edge_attr = self.bn_e_3(edge_attr)

        x_h,x_g,edge_attr,u = self.block_4(x_h,x_g,edge_index,edge_attr,u,batch_e,batch_h,batch_g)
        x_h = self.bn_xh_4(x_h)
        x_g = self.bn_xg_4(x_g)
        edge_attr = self.bn_e_4(edge_attr)

        x_h,x_g,edge_attr,u = self.block_last(x_h,x_g,edge_index,edge_attr,u,batch_e,batch_h,batch_g)

        time = torch.sum(Tmax*torch.sigmoid(edge_attr),dim=-1)

        if train:
            noise = self.noiselevel*(torch.rand(time.shape).float().cuda()-0.5)
            time = time+noise
            inttime = torch.floor(time)
            time = inttime + torch.sigmoid(self.sharpness*(time-0.5-inttime))
        else:
            time = torch.round(time)
        return time,edge_index
