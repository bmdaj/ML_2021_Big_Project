import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

import sys
sys.path.append("/Users/Pedro/Google Drive/MSc - Computational Physics/Applied Machine Learning/Big project/code/py_scripts")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import quantile_transform, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

from scipy.special import expit

from data_normalization import normalize_bands

#%% Load and transform data
path = "/Users/Pedro/Google Drive/MSc - Computational Physics/Applied Machine Learning/Big project/data/"
X = pd.read_csv(path + "frequencies_data.csv")
y = pd.read_csv(path + "params_data.csv")

N_BANDS = 5          # Number of bands
N_KPOINTS = 31       # Number of K points in a band
TEST_SIZE = 0.25     # Test size
RSTATE = 42          # Random state
T_TYPE = "quantile"  # Transformation type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RSTATE)

X_train = normalize_bands(X_train, 
                          num_bands = N_BANDS, 
                          num_k_points = N_KPOINTS,
                          transformation = T_TYPE)
X_test  = normalize_bands(X_test, 
                          num_bands = N_BANDS, 
                          num_k_points = N_KPOINTS,
                          transformation = T_TYPE)


class MyDataset(Dataset):    
    def __init__(self, X_data, y_data):
        self.input = X_data
        self.truth = y_data
        
    def __getitem__(self, index):
        return self.input[index], self.truth[index]
        
    def __len__ (self):
        return len(self.input)


train_data = MyDataset(torch.FloatTensor(np.float64(np.array(X_train))), 
                       torch.FloatTensor(np.float64(np.array(y_train))))
test_data = MyDataset(torch.FloatTensor(np.float64(np.array(X_test))), 
                       torch.FloatTensor(np.float64(np.array(y_test))))

#%% Data loaders
BATCH_SIZE = 150

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

#%% InverseNN model: from frequcnies to parameters
class InverseNN(nn.Module):
    def __init__(self, 
                 N_BLOCKS: int, 
                 D_IN: int, 
                 D_HIDDEN_BK: list, 
                 D_HIDDEN_FC: list,
                 D_OUT: int, 
                 P_DROPOUT: None):
        """
        Inverse NN, i.e. from frequencies to parameters:
        @param N_BLOCKS: number of input blocks, i.e., # of bands
        @param D_IN: number of inputs for each block
        @param D_HIDDEN_BK: list with the dimensions of each hidden layer in a block (BK)
        @param D_HIDDEN_FC: list with the dimensions of each hidden later in the FC
        @param D_OUT: dimension of the output layer in the FC, i.e., # of parameters
        @param P_DROPOUT: dropout probability
        """
        super(InverseNN, self).__init__()
        self.N_BLOCKS    = N_BLOCKS
        self.D_IN        = D_IN
        self.D_HIDDEN_BK = D_HIDDEN_BK
        self.D_HIDDEN_FC = D_HIDDEN_FC
        self.D_OUT       = D_OUT
        self.P_DROPOUT   = P_DROPOUT
        
        # Apply dropout?
        self.APPLY_DROPOUT = True
        if self.P_DROPOUT == None:
            self.APPLY_DROPOUT = False
            
        # Apply Batch Normalization?
        self.APPLY_BN = False
    
    
    def build_LinearBlock(self):
        """ Builds a single input block (there is one per band). """
        layers = []
        in_ = self.D_IN
        for D_H in self.D_HIDDEN_BK:
            layers.append(nn.Linear(in_, D_H))
            layers.append(nn.ReLU())
            if self.APPLY_DROPOUT: 
                layers.append(nn.Dropout(self.P_DROPOUT))
            if self.APPLY_BN:
                layers.append(nn.BatchNorm1d(D_H))
            in_ = D_H

        return nn.Sequential(*layers)
    
    
    def build_FC(self):
        """ Builds the Fully Connected bit. """
        layers = []
        in_ = self.N_BLOCKS * self.D_HIDDEN_BK[-1]
        for D_H in self.D_HIDDEN_FC:
            layers.append(nn.Linear(in_, D_H))
            layers.append(nn.ReLU())
            if self.APPLY_DROPOUT: 
                layers.append(nn.Dropout(self.P_DROPOUT))
            if self.APPLY_BN:
                layers.append(nn.BatchNorm1d(D_H))
            in_ = D_H
            
        layers.append(nn.Linear(in_, self.D_OUT))
            
        return nn.Sequential(*layers)
    
    
    @staticmethod
    def get_bands(tensor, step):
        for i in range(0, len(tensor), step):
            yield tensor[i:i+step]
    
    
    def forward_InverseSplitData(self, DATA):
        """ 
        Apply blocks to the input data:
        @param DATA: input data within a batch (BATCH_SIZE x D_IN)
        """
        fbands = torch.split(DATA, self.D_IN, dim=0)  # Frequency bands = columns
        assert(len(fbands) == self.N_BLOCKS)
        
        outputs  = []
        build_BK = self.build_LinearBlock()
        for fband in fbands:
            fband = fband.view(-1, fband.size(0))
            out_  = build_BK(fband)
            #out_ = out_.view(out_.size(0), -1)
            outputs.append(out_)
        
        return torch.cat(outputs, dim=1)
    
    
    def forward(self, DATA_BATCH):
        """ All the NN: Blocks + FC """
        out_ = []
        for SIM in DATA_BATCH:
            # SIM = SIMulation
            out_.append(self.forward_InverseSplitData(SIM))
        x = torch.tensor(out_[0])
        x = self.build_FC(x)
        
        return x  # Parameters
    


N_BLOCKS = N_BANDS
D_IN = N_KPOINTS
D_OUT = y.shape[1]
D_HIDDEN_BK = [D_IN, 100, 100]
D_HIDDEN_FC = [200, 200, 200]
P_DROPOUT = 0.25

our_model = InverseNN(N_BLOCKS, 
                      D_IN, 
                      D_HIDDEN_BK, 
                      D_HIDDEN_FC, 
                      D_OUT, 
                      P_DROPOUT)

# Let's test it
for batch, _ in train_loader:
    out_test = our_model(batch)
    break



#%% nothing 
class InverseNN(nn.Module):
    def __init__(self, 
                 N_BLOCKS: int, 
                 D_IN: int, 
                 D_HIDDEN_BK: list, 
                 D_HIDDEN_FC: list,
                 D_OUT: int, 
                 P_DROPOUT: None):
        """
        Inverse NN, i.e. from frequencies to parameters:
        @param N_BLOCKS: number of input blocks, i.e., # of bands
        @param D_IN: number of inputs for each block
        @param D_HIDDEN_BK: list with the dimensions of each hidden layer in a block (BK)
        @param D_HIDDEN_FC: list with the dimensions of each hidden later in the FC
        @param D_OUT: dimension of the output layer in the FC, i.e., # of parameters
        @param P_DROPOUT: dropout probability
        """
        """
        super(InverseNN, self).__init__()
        self.H_LAYERS_BK = nn.ModuleList()
        for D_H in D_HIDDEN_BK:
            self.H_LAYERS_BK.append()
        self.H_LAYERS_FC = nn.ModuleList()
        """
        

    
    def build_LinearBlock(self):
        """ Builds a single input block (there is one per band). """
        layers = []
        in_ = self.D_IN
        for D_H in self.D_HIDDEN_BK:
            layers.append(nn.Linear(in_, D_H))
            layers.append(nn.ReLU())
            if self.APPLY_DROPOUT: 
                layers.append(nn.Dropout(self.P_DROPOUT))
            if self.APPLY_BN:
                layers.append(nn.BatchNorm1d(D_H))
            in_ = D_H
            
        return nn.Sequential(*layers)
    
    
    def build_FC(self):
        """ Builds the Fully Connected bit. """
        layers = []
        in_ = self.N_BLOCKS * self.D_HIDDEN_BK[-1]
        for D_H in self.D_HIDDEN_FC:
            layers.append(nn.Linear(in_, D_H))
            layers.append(nn.ReLU())
            if self.APPLY_DROPOUT: 
                layers.append(nn.Dropout(self.P_DROPOUT))
            if self.APPLY_BN:
                layers.append(nn.BatchNorm1d(D_H))
            in_ = D_H
            
        layers.append(nn.Linear(in_, self.D_OUT))
            
        return nn.Sequential(*layers)
    
    
    def forward_InverseSplitData(self, SIM_DATA):
        """ 
        Apply blocks to the input data:
        @param DATA: input data within a batch (BATCH_SIZE x D_IN)
        """
        outputs  = []
        build_BK = self.build_LinearBlock()
        for fband in self.get_bands(SIM_DATA, self.D_IN):
            fband = fband.view(fband.size(0), -1)
            out_  = build_BK(fband)
            outputs.append(out_)
        
        return torch.cat(outputs, dim=1)
    

    @staticmethod
    def get_bands(tensor, step):
        for i in range(0, len(tensor), step):
            yield tensor[i:i+step]
    
    
    def forward(self, DATA_BATCH):
        """ All the NN: Blocks + FC """
        out_ = []
        for SIM in DATA_BATCH:
            # SIM = SIMulation
            out_.append(self.forward_InverseSplitData(SIM))
        x = torch.tensor(out_)
        x = self.build_FC(x)
        
        return x  # Parameters
