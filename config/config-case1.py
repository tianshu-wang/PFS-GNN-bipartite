import os
import lossfun1 
import lossfun2

case = 'case1'

# GNN structure parameters
n_x = 10
n_u = 10
n_h = 10
n_g = 14 

# Problem Parameters
Tmax = 15
datadir = 'graphs-case1/'
modeldir = 'models-case1/'
pre_modelname = 'model_gnn_pre.pth'
modelname = 'model_gnn.pth'
Loss = lossfun1.Lossfun

# Training Parameters
ntrain = 10 # number of training graphs
nvalid = 5 # number of validating graphs
ntest = 5 # number of testing graphs

train = False # True to train the model. 
nepoch_pre = 2000 # Pre-train
nepoch = 8000 # Gradually increase the strength of overtime punishment

lr_pre = 5e-4 #[1e-3,1e-3,5e-4,5e-4][idx]
penalty_pre = 1e-7 # time constraint strength in pre-training

lr = 5e-4 #[1e-3,1e-3,5e-4,5e-4][idx]
penalty_ini = 1e-7# initial
penalty_end = 1e-4

# Noisy Sigmoid Parameters
sharpness = 20
noiselevel = 0.3 #[0.2,0.3,0.2,0.3][idx]
