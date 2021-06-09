import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

import sys
sys.path.append("/Users/Pedro/Google Drive/MSc - Computational Physics/Applied Machine Learning/Big project/code/py_scripts")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
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
                 P_DROPOUT: None,
                 BOUNDS: list):
        """
        Inverse NN, i.e. from frequencies to parameters:
        @param N_BLOCKS: number of input blocks, i.e., # of bands
        @param D_IN: number of inputs for each block
        @param D_HIDDEN_BK: list with the dimensions of each hidden layer in a block (BK)
        @param D_HIDDEN_FC: list with the dimensions of each hidden later in the FC
        @param D_OUT: dimension of the output layer in the FC, i.e., # of parameters
        @param P_DROPOUT: dropout probability
        @param BOUNDS: list with lists containing the parameter bounds
        """
        super(InverseNN, self).__init__()
        self.D_IN = D_IN
        self.activation = nn.ReLU()
        self.bounds = BOUNDS
        
        # Hidden layers
        self.hidden_BK = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_BK[:-1], D_HIDDEN_BK[1:]):
            self.hidden_BK.append(nn.Linear(this_H, next_H))
            self.hidden_BK.append(self.activation)
            
        self.hidden_FC = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_FC[:-1], D_HIDDEN_FC[1:]):
            self.hidden_FC.append(nn.Linear(this_H, next_H))
            self.hidden_FC.append(self.activation)
            self.hidden_BK.append(nn.Dropout(P_DROPOUT))
        
        # Building models: Blocks and Fully Connected
        self.build_BK = nn.Sequential(
            nn.Linear(D_IN, D_HIDDEN_BK[0]),                        # Input layer (freqs)
            self.activation,                                        # Activation
            *self.hidden_BK)                                        # Hidden layers
        
        self.build_FC = nn.Sequential(
            nn.Linear(N_BLOCKS * D_HIDDEN_BK[-1], D_HIDDEN_FC[0]),  # Input layer
            self.activation,                                        # Activation
            *self.hidden_FC,                                        # Hidden layers
            nn.Linear(D_HIDDEN_FC[-1], D_OUT))                      # Output layer (params)
        
    
    @staticmethod
    def get_chunks(T, chunksize):
        for i in range(0, T.size(1), chunksize):
            yield T[:,i:i+chunksize]
    
    """        
    def bounded_output(self, x, lower, upper):
        #Activation function sepecific to each output to constrain it.
        scale = upper - lower
        return scale * F.sigmoid(x) + lower
    """
    
    def forward(self, X_IN):
        X_OUT = []
        for X_BAND in self.get_chunks(X_IN, self.D_IN):
            X_OUT.append(self.build_BK(X_BAND))
            
        X_OUT = torch.cat(X_OUT, dim=1)
        X_OUT = self.build_FC(X_OUT)
        """
        for i, bound in enumerate(self.bounds):
            lower, upper = bound[0], bound[1]
            X_OUT[:,i] = self.bounded_output(X_OUT[:,i], lower, upper)
        """
        return X_OUT
    

N_BLOCKS = N_BANDS
D_IN = N_KPOINTS
D_OUT = y.shape[1]
D_HIDDEN_BK = [D_IN, 100, 100]
D_HIDDEN_FC = [300, 700, 700, 700, 700]
P_DROPOUT = 0.25
BOUNDS = [[2., 20.], [0.1, 1.], [10., 130.], [0., 180.]]

# GPU support
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

our_Imodel = InverseNN(N_BLOCKS, 
                       D_IN, 
                       D_HIDDEN_BK, 
                       D_HIDDEN_FC, 
                       D_OUT, 
                       P_DROPOUT,
                       BOUNDS).to(device)

# Let's test it
TEST_IT = False
if TEST_IT:
    for batch, _ in train_loader:
        out_test = our_Imodel(batch)
        break


#%% Training
def train_model(model, criterion, optimizer, train_loader, valid_loader, scheduler, device, num_epochs):
    train_loss = []
    valid_loss = []
    
    for epoch in range(num_epochs):
        # Each epoch consits of a TRAINING and VALIDATION phase
        # TRAINING
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
            
        if scheduler == None: 
            pass
        else:
            scheduler.step()
            
        epoch_train_loss = running_loss/n_batches
        train_loss.append(epoch_train_loss)
        
        # VALIDATION
        model.eval()
        running_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                n_batches += 1
        
        epoch_valid_loss = running_loss/n_batches
        valid_loss.append(epoch_valid_loss)
        
        print('Epoch: {}/{} | Train loss: {:.4f} | Valid loss: {:.4f}'.format(
            epoch+1, num_epochs, epoch_train_loss, epoch_valid_loss))
    
    return model, train_loss, valid_loss

LEARNING_RATE = 0.01
EPOCHS = 50

criterion = nn.MSELoss()
optimizer = optim.Adam(our_Imodel.parameters(), lr=LEARNING_RATE)
step_lr_scheduler = None  #optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train the NN
model, train_loss, valid_loss = train_model(
    our_Imodel, 
    criterion,
    optimizer, 
    train_loader,
    test_loader,
    step_lr_scheduler, 
    device,
    EPOCHS)


#%% Predictions
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            
    return predictions

preds = predict(model, test_loader, device)

#%% Residual distribution
res = (y_test.to_numpy() - np.array(preds))/y_test.to_numpy()

NBINS = 100
XMIN, XMAX = -5., 5.
BW = (XMAX - XMIN)/NBINS
COLORS = ['k', 'tab:red', 'tab:green', 'tab:blue']

fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
for i in range(y.shape[1]):
    ax.hist(res[:,i], bins=NBINS, range=(XMIN, XMAX), color=COLORS[i],
            histtype='step', density=True, lw=1.25, label=y.columns[i]);
ax.legend()
ax.grid(True)

#%% Direct
class DirectNN(nn.Module):
    def __init__(self, 
                 D_IN: int, 
                 D_HIDDEN_FC: list,
                 D_OUT: int,
                 P_DROPOUT: None):
        """
        Direct NN, i.e. from parameters to frequencies:
        @param D_IN: number of inputs 
        @param D_HIDDEN_FC: list with the dimensions of each hidden later in the FC
        @param D_OUT: dimension of the output layer in the FC, i.e., # of bands * # k points
        @param P_DROPOUT: dropout probability
        """
        super(DirectNN, self).__init__()
        self.D_IN = D_IN
        self.activation = nn.ReLU()
        self.hidden_FC = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_FC[:-1], D_HIDDEN_FC[1:]):
            self.hidden_FC.append(nn.Linear(this_H, next_H))
            self.hidden_FC.append(self.activation)
        
        self.build_FC = nn.Sequential(
            nn.Linear(D_IN, D_HIDDEN_FC[0]),                    # Input layer
            self.activation,                                    # Activation
            *self.hidden_FC,                                    # Hidden layers
            nn.Linear(D_HIDDEN_FC[-1], D_OUT))                  # Output layer (frequencies)
    
    
    def forward(self, X_IN):
        X_OUT = self.build_FC(X_IN)
        
        return X_OUT

D_IN = 4
D_OUT = 155
D_HIDDEN_FC = [300, 700, 700, 700, 700]
P_DROPOUT = 0.25

our_Dmodel = DirectNN(D_IN,  
                      D_HIDDEN_FC, 
                      D_OUT, 
                      P_DROPOUT).to(device)

#%% Direct + Inverse 

def train_model(model_Inverse, model_Direct, criterion, optimizer, train_loader, valid_loader, scheduler, device, num_epochs):
    train_loss = []
    train_loss_Inverse = []
    train_loss_Direct = []
    valid_loss = []
    valid_loss_Inverse = []
    valid_loss_Direct = []
    
    for epoch in range(num_epochs):
        # Each epoch consits of a TRAINING and VALIDATION phase
        # TRAINING
        model_Inverse.train()
        model_Direct.train()
        running_loss = 0.0
        running_loss_Inverse = 0.0
        running_loss_Direct = 0.0
        n_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs_Inverse = model_Inverse(inputs)
            loss_Inverse = criterion(outputs_Inverse, labels)
            outputs_Direct = model_Direct(outputs_Inverse)
            loss_Direct = criterion(outputs_Direct, inputs)
            loss_Inverse.backward(retain_graph=True)
            loss_Direct.backward()
            optimizer.step()
            
            running_loss += loss_Inverse.item() + loss_Direct.item()
            running_loss_Inverse += loss_Inverse.item()
            running_loss_Direct += loss_Direct.item()
            n_batches += 1
            
        if scheduler == None: 
            pass
        else:
            scheduler.step()
            
        epoch_train_loss = running_loss/n_batches
        epoch_train_loss_Inverse = running_loss_Inverse/n_batches
        epoch_train_loss_Direct = running_loss_Direct/n_batches
        train_loss.append(epoch_train_loss)
        train_loss_Inverse.append(epoch_train_loss_Inverse)
        train_loss_Direct.append(epoch_train_loss_Direct)
        
        # VALIDATION
        model_Inverse.eval()
        model_Direct.eval()
        running_loss = 0.0
        running_loss_Inverse = 0.0
        running_loss_Direct  = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs_Inverse = model_Inverse(inputs)
                outputs_Direct  = model_Direct(outputs_Inverse)
                
                loss_Inverse = criterion(outputs_Inverse, labels)
                loss_Direct  = criterion(outputs_Direct, inputs)
                
                running_loss += loss_Inverse.item() + loss_Direct.item()
                running_loss_Inverse += loss_Inverse.item()
                running_loss_Direct += loss_Direct.item()
                n_batches += 1
        
        epoch_valid_loss = running_loss/n_batches
        epoch_valid_loss_Inverse = running_loss_Inverse/n_batches
        epoch_valid_loss_Direct = running_loss_Direct/n_batches
        valid_loss.append(epoch_valid_loss)
        valid_loss_Inverse.append(epoch_valid_loss_Inverse)
        valid_loss_Direct.append(epoch_valid_loss_Direct)
        
        print('Epoch: {}/{} | Total Train loss: {:.4f} | Total Valid loss: {:.4f}'.format(
            epoch+1, num_epochs, epoch_train_loss, epoch_valid_loss))
        print('Epoch: {}/{} | Inverse Train loss: {:.4f} | Inverse Valid loss:                             {:.4f}'.format(epoch+1, num_epochs, epoch_train_loss_Inverse,                                 epoch_valid_loss_Inverse))
        print('Epoch: {}/{} | Direct Train loss: {:.4f} | Direct Valid loss:                             {:.4f}'.format(epoch+1, num_epochs, epoch_train_loss_Direct,                                 epoch_valid_loss_Direct))
        print('-----------------------------------------------------------------')
    return model_Inverse, model_Direct, train_loss, valid_loss

# Train the NN
model, train_loss, valid_loss = train_model(
    our_Imodel,
    our_Dmodel, 
    criterion,
    optimizer, 
    train_loader,
    test_loader,
    step_lr_scheduler, 
    device,
    EPOCHS)