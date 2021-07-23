import torch
from torch_geometric.data import Data,DataLoader
from config import *
import gnn as g
# Load data
train_graphs=[]
for i in range(ntrain):
    igraph=torch.load(datadir+'graph-%d-train.pt'%i)
    train_graphs.append(igraph)
dataset = g.Loader(graphs_list=train_graphs)
dataloader = DataLoader(dataset,batch_size=1)
train_be = []
train_bh = []
train_bg = []
for i_batch,graph in enumerate(dataloader):
    # only correct for batchsize=1
    be = torch.zeros(graph.edge_attr.size(0),dtype=torch.long).cuda()
    bh = torch.zeros(graph.x_h.size(0),dtype=torch.long).cuda()
    bg = torch.zeros(graph.x_g.size(0),dtype=torch.long).cuda()
    train_be.append(be)
    train_bh.append(bh)
    train_bg.append(bg)
valid_graphs = []
for igraph in range(nvalid):
    i_valid_graph = torch.load(datadir+'graph-%d-valid.pt'%igraph)
    valid_graphs.append(i_valid_graph)
valid_set = g.Loader(graphs_list=valid_graphs)
validloader = DataLoader(valid_set,batch_size=1)
valid_be = []
valid_bh = []
valid_bg = []
for i_batch,graph in enumerate(validloader):
    # only correct for batchsize=1
    be = torch.zeros(graph.edge_attr.size(0),dtype=torch.long).cuda()
    bh = torch.zeros(graph.x_h.size(0),dtype=torch.long).cuda()
    bg = torch.zeros(graph.x_g.size(0),dtype=torch.long).cuda()
    valid_be.append(be)
    valid_bh.append(bh)
    valid_bg.append(bg)
test_graphs = []
for igraph in range(ntest):
    i_test_graph = torch.load(datadir+'graph-%d-test.pt'%(igraph))
    test_graphs.append(i_test_graph)
test_set = g.Loader(graphs_list=test_graphs)
testloader = DataLoader(test_set,batch_size=1)
test_be = []
test_bh = []
test_bg = []
for i_batch,graph in enumerate(testloader):
    # only correct for batchsize=1
    be = torch.zeros(graph.edge_attr.size(0),dtype=torch.long).cuda()
    bh = torch.zeros(graph.x_h.size(0),dtype=torch.long).cuda()
    bg = torch.zeros(graph.x_g.size(0),dtype=torch.long).cuda()
    test_be.append(be)
    test_bh.append(bh)
    test_bg.append(bg)
