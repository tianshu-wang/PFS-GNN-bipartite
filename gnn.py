import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean,scatter
from torch_geometric.data import Data,Dataset,DataLoader
from torch_geometric.nn import MessagePassing
import warnings
warnings.filterwarnings("ignore")

F_e = 10
F_u = 10

F_xs = 10
F_xt = 5

F_e_out = 5


class BipartiteData(Data):
    def __init__(self, edge_index, x_s, x_t, edge_attr,u):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index.cuda()
        self.x_s = x_s.cuda()
        self.x_t = x_t.cuda()
        self.edge_attr = edge_attr.cuda()
        self.u = u.cuda()
        self.num_nodes = len(self.x_t) # useless, just to avoid warnings
    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
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
        self.edge_mlp = Seq(Lin(F_xs+F_xt+F_e+F_u,F_e), LeakyReLU(0.1), Lin(F_e,F_e))
    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_e):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        src, tgt = edge_index
        out = torch.cat([x_s[src], x_t[tgt], edge_attr, u[batch_e]], 1)
        out =  self.edge_mlp(out)
        return out
class EdgeModel_out(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_out, self).__init__()
        self.edge_mlp = Seq(Lin(F_xs+F_xt+F_e+F_u,F_e_out), LeakyReLU(0.1), Lin(F_e_out,F_e_out))
    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_e):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        src, tgt = edge_index
        out = torch.cat([x_s[src], x_t[tgt], edge_attr, u[batch_e]], 1)
        out =  self.edge_mlp(out)
        return out
class SModel(torch.nn.Module):
    def __init__(self):
        super(SModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(F_e+F_xt,F_e+F_xt), LeakyReLU(0.1), Lin(F_e+F_xt,F_e+F_xt))
        self.node_mlp_2 = Seq(Lin(F_xs+1+4*(F_e+F_xt)+F_u,F_xs), LeakyReLU(0.1), Lin(F_xs,F_xs))
    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_s):
        # x: [N, F_xs], where N is the number of S-nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        src, tgt = edge_index
        out = torch.cat([x_t[tgt],edge_attr], dim=1)
        out = self.node_mlp_1(out)
        ns = torch.ones(len(out),1).float().cuda() 
        n = scatter(ns, src, dim=0, dim_size=x_s.size(0),reduce='sum')
        a = scatter(out, src, dim=0, dim_size=x_s.size(0),reduce='mean') # mu
        b = torch.sqrt(1e-6+F.relu(scatter(out**2, src, dim=0, dim_size=x_s.size(0),reduce='mean')-a**2)) # sigma
        c = scatter((out-a[src])**3, src, dim=0, dim_size=x_s.size(0),reduce='mean')/b**3 #skewness
        d = scatter((out-a[src])**4, src, dim=0, dim_size=x_s.size(0),reduce='mean')/b**4 #kurtosis
        out = torch.cat([x_s,n,a,b,c,d,u[batch_s]], dim=1)
        out = self.node_mlp_2(out)
        return out
class TModel(torch.nn.Module):
    def __init__(self):
        super(TModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(F_e+F_xs,F_e+F_xs), LeakyReLU(0.1), Lin(F_e+F_xs,F_e+F_xs))
        self.node_mlp_2 = Seq(Lin(F_xt+1*(F_e+F_xs)+F_u,F_xt), LeakyReLU(0.1), Lin(F_xt,F_xt))
    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_t):
        # x: [N, F_xs], where N is the number of S-nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        src, tgt = edge_index
        out = torch.cat([x_s[src],edge_attr], dim=1)
        out = self.node_mlp_1(out)
        ns = torch.ones(len(out),1).float().cuda() 
        #n = scatter(ns, tgt, dim=0, dim_size=x_t.size(0),reduce='sum')
        a = scatter(out, tgt, dim=0, dim_size=x_t.size(0),reduce='sum') # mu
        #b = torch.sqrt(1e-6+F.relu(scatter(out**2, tgt, dim=0, dim_size=x_t.size(0),reduce='mean')-a**2)) # sigma
        #c = scatter((out-a[tgt])**3, tgt, dim=0, dim_size=x_t.size(0),reduce='mean')/b**3 #skewness
        #d = scatter((out-a[tgt])**4, tgt, dim=0, dim_size=x_t.size(0),reduce='mean')/b**4 #kurtosis
        out = torch.cat([x_t,a,u[batch_t]], dim=1)
        out = self.node_mlp_2(out)
        return out
class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(F_u+F_xs+F_xt,F_u), LeakyReLU(0.1), Lin(F_u, F_u))
    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_s,batch_t):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x_s,batch_s,dim=0),scatter_mean(x_t,batch_t,dim=0)], dim=1)
        return self.global_mlp(out)
class Block(torch.nn.Module):
    def __init__(self,edge_model=None,s_model=None,t_model=None,u_model=None):
        super(Block, self).__init__()
        self.edge_model = edge_model
        self.s_model = s_model 
        self.t_model = t_model
        self.global_model = u_model
    def forward(self,x_s,x_t,edge_index,edge_attr,u,batch_e,batch_s,batch_t):
        if self.edge_model is not None:
            edge_attr = self.edge_model(x_s, x_t, edge_index, edge_attr, u, batch_e)
        if self.s_model is not None:
            x_s = self.s_model(x_s, x_t, edge_index, edge_attr, u, batch_s)
        if self.t_model is not None:
            x_t = self.t_model(x_s, x_t, edge_index, edge_attr, u, batch_t)
        if self.global_model is not None:
            u = self.global_model(x_s, x_t, edge_index, edge_attr, u, batch_s,batch_t)
        return x_s,x_t,edge_attr,u
class GNN(torch.nn.Module):
    def __init__(self,gumbel=False):
        super(GNN,self).__init__()
        self.gumbel = gumbel
        self.tau = 0.1
        self.hard = True
        self.classes = torch.arange(F_e_out).float().cuda()
        self.block_1 = Block(EdgeModel(),SModel(),TModel(),GlobalModel())
        self.bn_xs_1 = nn.BatchNorm1d(F_xs)
        self.bn_xt_1 = nn.BatchNorm1d(F_xt)
        self.bn_e_1 = nn.BatchNorm1d(F_e)
        self.block_2 = Block(EdgeModel(),SModel(),TModel(),GlobalModel())
        self.bn_xs_2 = nn.BatchNorm1d(F_xs)
        self.bn_xt_2 = nn.BatchNorm1d(F_xt)
        self.bn_e_2 = nn.BatchNorm1d(F_e)
        self.block_3 = Block(EdgeModel(),SModel(),TModel(),GlobalModel())
        self.bn_xs_3 = nn.BatchNorm1d(F_xs)
        self.bn_xt_3 = nn.BatchNorm1d(F_xt)
        self.bn_e_3= nn.BatchNorm1d(F_e)
        self.block_last = Block(EdgeModel_out())
    def forward(self,data,batch_e,batch_s,batch_t):
        x_s = data.x_s
        x_t = data.x_t
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u


        x_s,x_t,edge_attr,u = self.block_1(x_s,x_t,edge_index,edge_attr,u,batch_e,batch_s,batch_t)
        x_s = self.bn_xs_1(x_s)
        x_t = self.bn_xt_1(x_t)
        edge_attr = self.bn_e_1(edge_attr)
        x_s,x_t,edge_attr,u = self.block_2(x_s,x_t,edge_index,edge_attr,u,batch_e,batch_s,batch_t)
        x_s = self.bn_xs_2(x_s)
        x_t = self.bn_xt_2(x_t)
        edge_attr = self.bn_e_2(edge_attr)
        x_s,x_t,edge_attr,u = self.block_3(x_s,x_t,edge_index,edge_attr,u,batch_e,batch_s,batch_t)
        x_s = self.bn_xs_3(x_s)
        x_t = self.bn_xt_3(x_t)
        edge_attr = self.bn_e_3(edge_attr)
        x_s,x_t,edge_attr,u = self.block_last(x_s,x_t,edge_index,edge_attr,u,batch_e,batch_s,batch_t)

        if self.gumbel:
            prob = F.gumbel_softmax(edge_attr,tau=self.tau,hard=self.hard)
            time = torch.sum(prob*self.classes,dim=-1)
            extra_loss = 0.
        else:
            softmax = nn.Softmax(dim=-1)
            prob = softmax(edge_attr)
            time = torch.sum(prob*self.classes,dim=-1)
            temp = self.classes.view(1,5).repeat(len(time),1)
            extra_loss = torch.sum((temp-time.view(-1,1))**2*prob) # std of time
        return time,edge_index,extra_loss,edge_attr
