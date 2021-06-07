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
        self.D_IN = D_IN
        self.activation = nn.ReLU()
        
        # Hidden layers
        self.hidden_BK = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_BK[:-1], D_HIDDEN_BK[1:]):
            self.hidden_BK.append(nn.Linear(this_H, next_H))
            self.hidden_BK.append(self.activation)
            
        self.hidden_FC = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_FC[:-1], D_HIDDEN_FC[1:]):
            self.hidden_FC.append(nn.Linear(this_H, next_H))
            self.hidden_FC.append(self.activation)
        
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
    
        
    def forward(self, X_IN):
        X_OUT = []
        for X_BAND in self.get_chunks(X_IN, self.D_IN):
            X_OUT.append(self.build_BK(X_BAND))
            
        X_OUT = torch.cat(X_OUT, dim=1)
        X_OUT = self.build_FC(X_OUT)
        
        return X_OUT
    

N_BLOCKS = N_BANDS
D_IN = N_KPOINTS
D_OUT = y.shape[1]
D_HIDDEN_BK = [D_IN, 100, 100]
D_HIDDEN_FC = [200, 200, 200]
P_DROPOUT = 0.25

# GPU support
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

our_model = InverseNN(N_BLOCKS, 
                      D_IN, 
                      D_HIDDEN_BK, 
                      D_HIDDEN_FC, 
                      D_OUT, 
                      P_DROPOUT).to(device)

# Let's test it
TEST_IT = False
if TEST_IT:
    for batch, _ in train_loader:
        out_test = our_model(batch)
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

LEARNING_RATE = 0.05
EPOCHS = 5

criterion = nn.MSELoss()
optimizer = optim.Adam(our_model.parameters(), lr=LEARNING_RATE)
step_lr_scheduler = None  #optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train the NN
model, train_loss, valid_loss = train_model(
    our_model, 
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
