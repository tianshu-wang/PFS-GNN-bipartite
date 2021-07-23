import os
import lossfun1 
import lossfun2

case = 'case2'

# GNN structure parameters
n_x = 10
n_u = 10
n_h = 10
n_g = 5

# Problem Parameters
Tmax = 4
datadir = 'graphs-case2/'
modeldir = 'models-case2/'
pre_modelname = 'model_gnn_pre.pth'
modelname = 'model_gnn.pth'
Loss = lossfun2.Lossfun

# Training Parameters
ntrain = 10 # number of training graphs
nvalid = 10 # number of validating graphs
ntest = 5 # number of testing graphs

train = False # True to train the model. 
nepoch_pre = 2000 # Pre-train
nepoch = 8000 # Gradually increase the strength of overtime punishment

lr_pre = 1e-3 #[1e-3,1e-3,5e-4,5e-4][idx]
penalty_pre = 1e-1 # time constraint strength in pre-training

lr = 1e-3 #[1e-3,1e-3,5e-4,5e-4][idx]
penalty_ini = 1e-1# initial
penalty_end = 1e-0

# Noisy Sigmoid Parameters
sharpness = 20
noiselevel = 0.3 #[0.2,0.3,0.2,0.3][idx]
