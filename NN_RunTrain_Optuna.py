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

from data_normalization import normalize_bands, transform_parameters, antitransform_parameters, unormalize_bands
from loss_functions import LogCoshLoss, XTanhLoss, XSigmoidLoss

import optuna
from optuna.trial import TrialState

#%% Load and transform data
path = "/Users/Pedro/Google Drive/MSc - Computational Physics/Applied Machine Learning/Big project/data/"
X = pd.read_csv(path + "frequencies_data.csv")
y = pd.read_csv(path + "params_data.csv")
#y = y[y.columns[1:]]

N_BANDS = 5          # Number of bands
N_KPOINTS = 31       # Number of K points in a band
TEST_SIZE = 0.25     # Test size
RSTATE = 42          # Random state
T_TYPE = "quantile"  # Transformation type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RSTATE)

X_train, tfreqs = normalize_bands(
    X_train, 
    num_bands = N_BANDS, 
    num_k_points = N_KPOINTS,
    transformation = T_TYPE
)

for i in range(N_BANDS):
    column_start = "Band_" + str(i) + "_k_0"
    column_end   = "Band_" + str(i) + "_k_" + str(N_KPOINTS-1)
    
    df_bands = X_test.loc[:, column_start:column_end]
    columns  = df_bands.columns
    
    tfreq = tfreqs[i]
    df_bands = tfreq.transform(df_bands)
    df_bands = pd.DataFrame(df_bands, columns=columns)
    
    if i == 0:
        X_test_T = df_bands.copy()
    else:
        X_test_T = pd.concat([X_test_T, df_bands], axis=1)

X_test = X_test_T
NORM_FLAG = True

"""    
X_test, tfreqs_test  = normalize_bands(
    X_test, 
    num_bands = N_BANDS, 
    num_k_points = N_KPOINTS,
    transformation = T_TYPE
)
"""

TRANS_PARAMS = False
if TRANS_PARAMS:
    y_train, t_train = transform_parameters(y_train, T_TYPE)
    y_test,  t_test  = transform_parameters(y_test, T_TYPE)


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
                 P_DROPOUT: float,
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
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(P_DROPOUT)
        self.bounds = BOUNDS
        
        # Apply Batch Normalization?
        self.APPLY_BN = True
        
        # Hidden layers
        self.hidden_BK = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_BK[:-1], D_HIDDEN_BK[1:]):
            self.hidden_BK.append(nn.Linear(this_H, next_H))
            self.hidden_BK.append(self.activation)
            
        self.hidden_FC = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_FC[:-1], D_HIDDEN_FC[1:]):
            self.hidden_FC.append(nn.Linear(this_H, next_H))
            self.hidden_FC.append(self.activation)
            self.hidden_FC.append(self.dropout)
            if self.APPLY_BN: self.hidden_FC.append(nn.BatchNorm1d(next_H))
            
        
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
D_HIDDEN_FC = [500, 2000, 2000, 2000, 2000]
P_DROPOUT = 0.15
BOUNDS = [[2., 20.], [0.1, 1.], [0.01, 150.], [0., 180.]]

# GPU support
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

our_Imodel = InverseNN(N_BLOCKS, 
                       D_IN, 
                       D_HIDDEN_BK, 
                       D_HIDDEN_FC, 
                       D_OUT,
                       P_DROPOUT,
                       BOUNDS)
our_Imodel.APPLY_BN = False

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# Xavier initialization of the weights
INI_WTS = True
if INI_WTS: our_Imodel.apply(init_weights).to(device)

# Let's test it
TEST_IT = True
if TEST_IT:
    for batch, _ in train_loader:
        out_test = our_Imodel(batch)
        break


#%% Training
TRAIN_INV = False
if TRAIN_INV:
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
    
    LEARNING_RATE = 0.001
    EPOCHS = 5
    
    criterion = nn.MSELoss()
    optimizer_I = optim.Adam(our_Imodel.parameters(), lr=LEARNING_RATE)
    # Other criteria:
    # - nn.SmoothL1Loss
    
    STEP_SIZE = 3
    GAMMA = 0.1
    step_lr_scheduler = None  #optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Train the NN
    model, train_loss, valid_loss = train_model(
        our_Imodel, 
        criterion,
        optimizer_I, 
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

if TRAIN_INV:
    y_pred = pd.DataFrame(np.array(predict(model, test_loader, device)))
    if TRANS_PARAMS:
        y_pred = antitransform_parameters(y_pred, t_test)
        y_test = antitransform_parameters(y_test, t_test)
        y_test.columns = y.columns

#%% Plot of the residuals
if TRAIN_INV:
    res = (y_test.to_numpy() - y_pred.to_numpy())/y_test.to_numpy()
    
    NBINS = 100
    XMIN, XMAX = -50., 50.
    BW = (XMAX - XMIN)/NBINS
    COLORS = ['k', 'tab:red', 'tab:green', 'tab:blue']
    
    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
    for i in range(y.shape[1]):
        ax.hist(res[:,i], bins=NBINS, range=(XMIN, XMAX), color=COLORS[i],
                histtype='step', density=True, lw=1.25, label=y.columns[i]);
    ax.legend()
    ax.grid(True)
    

#%% Plot of the parameters
if TRAIN_INV:
    NBINS = 100
    COLORS = ['k', 'r']
    
    fig, axs = plt.subplots(2, 2, figsize=(12,8), tight_layout=True)
    axs = np.ravel(axs)
    for i in range(y.shape[1]):
        col = y.columns[i]
        axs[i].hist(y_test.loc[:,col].to_numpy(), bins=NBINS, color='k',
                    histtype='step', density=True, lw=1.25, label='Truth',
                    facecolor='gray', fill=True, alpha=0.5)
        axs[i].hist(y_pred.loc[:,i].to_numpy(), bins=NBINS, color='r',
                    histtype='step', density=True, lw=1.25, label='Prediction')
        
        axs[i].set(xlabel = y.columns[i]);
        axs[i].legend();

#%% Direct
class DirectNN(nn.Module):
    def __init__(self, 
                 D_IN: int, 
                 D_HIDDEN_FC: list,
                 D_OUT: int,
                 P_DROPOUT: float):
        """
        Direct NN, i.e. from parameters to frequencies:
        @param D_IN: number of inputs 
        @param D_HIDDEN_FC: list with the dimensions of each hidden later in the FC
        @param D_OUT: dimension of the output layer in the FC, i.e., # of bands * # k points
        @param P_DROPOUT: dropout probability
        """
        super(DirectNN, self).__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(P_DROPOUT)
        self.APPLY_BN = True
        
        self.hidden_FC = nn.ModuleList()
        for this_H, next_H in zip(D_HIDDEN_FC[:-1], D_HIDDEN_FC[1:]):
            self.hidden_FC.append(nn.Linear(this_H, next_H))
            self.hidden_FC.append(self.activation)
            self.hidden_FC.append(self.dropout)
            if self.APPLY_BN: self.hidden_FC.append(nn.BatchNorm1d(next_H))
        
        self.build_FC = nn.Sequential(
            nn.Linear(D_IN, D_HIDDEN_FC[0]),                    # Input layer
            self.activation,                                    # Activation
            *self.hidden_FC,                                    # Hidden layers
            nn.Linear(D_HIDDEN_FC[-1], D_OUT))                  # Output layer (frequencies)
            
    
    def forward(self, X_IN):
        X_OUT = self.build_FC(X_IN)
        
        return X_OUT

D_IN = y.shape[1]
D_OUT = 155
D_HIDDEN_FC = [300, 1000, 1000, 1000]
P_DROPOUT = 0.15

our_Dmodel = DirectNN(D_IN,  
                      D_HIDDEN_FC, 
                      D_OUT, 
                      P_DROPOUT)
our_Dmodel.APPLY_BN = False

# Xavier initialization of the weights
INI_WTS = True
if INI_WTS: our_Dmodel.apply(init_weights).to(device)

# Let's test it
TEST_IT = True
if TEST_IT:
    for _, batch in train_loader:
        out_test = our_Dmodel(batch)
        break
    
#%% Direct + Inverse 
def train_model(model_I, 
                model_D, 
                criterion,
                optimizer_I, 
                optimizer_D,
                train_loader, 
                valid_loader, 
                scheduler, 
                device, 
                num_epochs):
    
    train_loss = []
    train_loss_I, train_loss_D = [], []
    
    valid_loss = []
    valid_loss_I, valid_loss_D = [], []
    
    for epoch in range(num_epochs):
        # TRAINING
        model_I.train(), model_D.train()
        running_loss = 0.0
        running_loss_I = 0.0
        running_loss_D = 0.0
        n_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer_I.zero_grad()
            optimizer_D.zero_grad()
            
            outputs_I = model_I(inputs)
            outputs_D = model_D(outputs_I)
            loss_I = criterion(outputs_I, labels)
            loss_D = criterion(outputs_D, inputs)
            loss_I.backward(retain_graph=True)
            loss_D.backward()
            optimizer_I.step()
            optimizer_D.step()
            
            running_loss   += loss_I.item() + loss_D.item()
            running_loss_I += loss_I.item()
            running_loss_D += loss_D.item()
            n_batches += 1
            
        if scheduler == None: 
            pass
        else:
            scheduler[0].step()  # InverseNN
            scheduler[1].step()  # DirectNN
            
        epoch_train_loss = running_loss/n_batches
        epoch_train_loss_I = running_loss_I/n_batches
        epoch_train_loss_D = running_loss_D/n_batches
        
        train_loss.append(epoch_train_loss)
        train_loss_I.append(epoch_train_loss_I)
        train_loss_D.append(epoch_train_loss_D)
        
        # VALIDATION
        model_I.eval(), model_D.eval()
        running_loss = 0.0
        running_loss_I = 0.0
        running_loss_D = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs_I = model_I(inputs)
                outputs_D = model_D(outputs_I)
                
                loss_I = criterion(outputs_I, labels)
                loss_D = criterion(outputs_D, inputs)
                
                running_loss += loss_I.item() + loss_D.item()
                running_loss_I += loss_I.item()
                running_loss_D += loss_D.item()
                n_batches += 1
        
        epoch_valid_loss = running_loss/n_batches
        epoch_valid_loss_I = running_loss_I/n_batches
        epoch_valid_loss_D = running_loss_D/n_batches
        
        valid_loss.append(epoch_valid_loss)
        valid_loss_I.append(epoch_valid_loss_I)
        valid_loss_D.append(epoch_valid_loss_D)
        
        print('Epoch: {}/{} | Total Train loss: {:.4f}   | Total Valid loss: {:.4f}'.format(
            epoch+1, num_epochs, epoch_train_loss, epoch_valid_loss))
        print('Epoch: {}/{} | Inverse Train loss: {:.4f} | Inverse Valid loss: {:.4f}'.format(
            epoch+1, num_epochs, epoch_train_loss_I, epoch_valid_loss_I))
        print('Epoch: {}/{} | Direct Train loss: {:.4f}  | Direct Valid loss: {:.4f}'.format(
            epoch+1, num_epochs, epoch_train_loss_D, epoch_valid_loss_D))
        print('-----------------------------------------------------------------------')
    
    return model_I, model_D, train_loss, valid_loss


BATCH_SIZE = 150
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset=test_data,  batch_size=BATCH_SIZE, shuffle=False)

LEARNING_RATE = 0.001
EPOCHS = 150

criterion = nn.MSELoss()
optimizer_I = optim.Adamax(our_Imodel.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adamax(our_Dmodel.parameters(), lr=LEARNING_RATE)
# Other optimizers
# - SGD w/ momentum
# - Adam
# - Adagrad
# - Adadelta
# - Adamax

STEP_SIZE = 5
GAMMA = 0.01
lr_scheduler_I = optim.lr_scheduler.StepLR(optimizer_I, step_size=STEP_SIZE, gamma=GAMMA)
lr_scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=STEP_SIZE, gamma=GAMMA)
lr_scheduler = None  #[lr_scheduler_I, lr_scheduler_D]

# Train the NN
model_I, model_D, train_loss, valid_loss = train_model(
    our_Imodel,
    our_Dmodel,
    criterion,
    optimizer_I,
    optimizer_D,
    train_loader,
    test_loader,
    lr_scheduler, 
    device,
    EPOCHS)

#%% Predictions
p_pred = pd.DataFrame(np.array(predict(model_I, test_loader, device)))

if TRANS_PARAMS:
    p_pred = antitransform_parameters(p_pred, t_test)
    y_test = antitransform_parameters(y_test, t_test)
    y_test.columns = y.columns
    
    
def predict_direct(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, labels in data_loader:
            labels = labels.to(device)
            outputs = model(labels)
            predictions.extend(outputs.numpy())
            
    return predictions

f_pred = pd.DataFrame(np.array(predict_direct(model_D, test_loader, device)))
f_pred.columns = X.columns
f_pred = unormalize_bands(f_pred, tfreqs)
if NORM_FLAG == True:
    X_test = unormalize_bands(X_test, tfreqs)
    NORM_FLAG = False

#%% Plot of residuals
res = np.abs(y_test.to_numpy() - p_pred.to_numpy())/y_test.to_numpy()

NBINS = 100
XMIN, XMAX = -50., 50.
BW = (XMAX - XMIN)/NBINS
COLORS = ['k', 'tab:red', 'tab:green', 'tab:blue']

fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
for i in range(y.shape[1]):
    ax.hist(res[:,i], bins=NBINS, range=(XMIN, XMAX), color=COLORS[i],
            histtype='step', density=True, lw=1.25, label=y.columns[i]);
ax.legend()
ax.grid(True)
#ax.set_xscale('log')


#%% Predictions of the parameters
NBINS = 75
COLORS = ['k', 'r']

fig, axs = plt.subplots(2, 2, figsize=(12,6), tight_layout=True)
axs = np.ravel(axs)
for i in range(y.shape[1]):
    col = y.columns[i]
    axs[i].hist(y_test.loc[:,col].to_numpy(), bins=NBINS, color='k',
                histtype='step', density=True, lw=1.25, label='Truth',
                facecolor='gray', fill=True, alpha=0.5)
    axs[i].hist(p_pred.loc[:,i].to_numpy(), bins=NBINS, color='r',
                histtype='step', density=True, lw=1.25, label='Prediction')
    
    axs[i].set(xlabel = y.columns[i]);
    axs[i].legend();

fig.savefig(path + '../imgs/ParamsDist_' + 
            str(criterion)[:-2] + '_' + 
            str(optimizer_I).split()[0] + '.pdf')


#%% Predictions of the frequencies
COLORS = ['k', 'r', 'tab:green', 'tab:blue', 'tab:orange']
fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
for i in range(N_BANDS):
    ax.plot(f_pred.iloc[15,i*31:(i+1)*31].values, color=COLORS[i])
    ax.plot(X_test.iloc[15,i*31:(i+1)*31].values, '--', color=COLORS[i])
ax.set(xlabel='$k$ points', ylabel='$\omega$')

fig.savefig(path + '../imgs/FreqBands_' + 
            str(criterion)[:-2] + '_' + 
            str(optimizer_I).split()[0] + '.pdf')

#%% HP optimization with Optuna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RSTATE)
X_train.index, y_train.index = np.arange(X_train.shape[0]), np.arange(y_train.shape[0])
X_test.index,  y_test.index  = np.arange(X_test.shape[0]),  np.arange(y_test.shape[0])

X_train, tfreqs = normalize_bands(
    X_train, 
    num_bands = N_BANDS, 
    num_k_points = N_KPOINTS,
    transformation = T_TYPE
)

for i in range(N_BANDS):
    column_start = "Band_" + str(i) + "_k_0"
    column_end   = "Band_" + str(i) + "_k_" + str(N_KPOINTS-1)
    
    df_bands = X_test.loc[:, column_start:column_end]
    columns  = df_bands.columns
    
    tfreq = tfreqs[i]
    df_bands = tfreq.transform(df_bands)
    df_bands = pd.DataFrame(df_bands, columns=columns)
    
    if i == 0:
        X_test_T = df_bands.copy()
    else:
        X_test_T = pd.concat([X_test_T, df_bands], axis=1)

X_test = X_test_T

TRANS_PARAMS = False
if TRANS_PARAMS:
    y_train, t_train = transform_parameters(y_train, T_TYPE)
    y_test,  t_test  = transform_parameters(y_test, T_TYPE)
    
    
def define_models(trial, 
                  N_BLOCKS, D_IN_Inv, D_OUT_Inv,  # Inverse-wise
                  D_IN_Dir, D_OUT_Dir,            # Direct-wise
                  P_DROPOUT, BOUNDS):
    """
    Define of the model based on Optuna trials.
    @param N_BLOCKS: number of bands/input blocks
    @param D_IN_{}: input size of Inverse/Direct
    @param D_OUT_{}: output size of Inverse/Direct
    @param P_DROPOUT: dropout probability
    @param BOUNDS: bounds for the parameters
    @hyperpar n_layers: number of layers
    @hyperpar n_neuron: size of each layer
    """
    n_layers_IBK = trial.suggest_int('n_layers_IBK', 2, 5)  # Inverse BK
    n_layers_IFC = trial.suggest_int('n_layers_IFC', 4, 7)  # Inverse FC
    n_layers_DFC = trial.suggest_int('n_layers_DFC', 3, 7)  # Direct FC
    
    layers_IBK, layers_IFC, layers_DFC = [], [], []
    
    for i in range(n_layers_IBK):
        layers_IBK.append(trial.suggest_int('DH_IBK_{}'.format(0), 50, 150))
        
    for i in range(n_layers_IFC):
        layers_IFC.append(trial.suggest_int('DH_IFC_{}'.format(0), 500, 2500))
    
    for i in range(n_layers_DFC):
        layers_DFC.append(trial.suggest_int('DH_DFC_{}'.format(0), 300, 1300))
    
    print(layers_IBK)
    print(layers_IFC)
    
    model_I = InverseNN(N_BLOCKS, D_IN_Inv, layers_IBK, layers_IFC, D_OUT_Inv, P_DROPOUT, BOUNDS).to(device)
    model_D = DirectNN(D_IN_Dir, layers_DFC, D_OUT_Dir, P_DROPOUT).to(device)
    
    return model_I, model_D


def train_model_pruning(trial, 
                        model_I, 
                        model_D, 
                        criterion,
                        optimizer_I, 
                        optimizer_D,
                        train_loader, 
                        valid_loader, 
                        scheduler, 
                        device, 
                        num_epochs):
    """" Model training taking into account Optuna pruning method """
    valid_loss = []
    
    for epoch in range(num_epochs):
        # TRAINING
        model_I.train(), model_D.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer_I.zero_grad()
            optimizer_D.zero_grad()
            
            outputs_I = model_I(inputs)
            outputs_D = model_D(outputs_I)
            loss_I = criterion(outputs_I, labels)
            loss_D = criterion(outputs_D, inputs)
            loss_I.backward(retain_graph=True)
            loss_D.backward()
            optimizer_I.step()
            optimizer_D.step()
            
        if scheduler == None: 
            pass
        else:
            scheduler[0].step()  # InverseNN
            scheduler[1].step()  # DirectNN
        
        # VALIDATION
        model_I.eval(), model_D.eval()
        running_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs_I = model_I(inputs)
                outputs_D = model_D(outputs_I)
                
                loss_I = criterion(outputs_I, labels)
                loss_D = criterion(outputs_D, inputs)
                
                running_loss += loss_I.item() + loss_D.item()
                n_batches += 1
        
        epoch_valid_loss = running_loss/n_batches
        valid_loss.append(epoch_valid_loss)
        
        trial.report(epoch_valid_loss, epoch)
    
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return valid_loss, trial


def objective(trial):
    """ Objective function to be minimized by Optuna """
    N_BLOCKS, D_IN_Inv, D_OUT_Inv = 5, 155, 4
    D_IN_Dir, D_OUT_Dir = 4, 155
    P_DROPOUT = 0.15
    BOUNDS = [[2., 20.], [0.1, 1.], [0.01, 150.], [0., 180.]]  # not necessary here
    
    our_Imodel, our_Dmodel = define_models(trial, 
                                           N_BLOCKS, D_IN_Inv, D_OUT_Inv, 
                                           D_IN_Dir, D_OUT_Dir,
                                           P_DROPOUT, BOUNDS)
    
    our_Imodel.APPLY_BN = False
    our_Dmodel.APPLY_BN = False
    
    # Xavier initialization of the weights
    INI_WTS = False
    if INI_WTS: 
        our_Imodel.apply(init_weights).to(device)
        our_Dmodel.apply(init_weights).to(device)
    
    # We don't use the whole dataset for HP optim., so we sample it
    # NB: make sure it is transformed!
    frac = 0.05
    TRAIN_SAMPLE_SIZE = int(frac*X_train.shape[0])
    TEST_SAMPLE_SIZE  = int(frac*X_test.shape[0])
    
    sample_train = np.random.choice(X_train.shape[0], TRAIN_SAMPLE_SIZE)
    sample_test  = np.random.choice(X_test.shape[0], TEST_SAMPLE_SIZE)
    SX_train, Sy_train = X_train.loc[sample_train,:], y_train.loc[sample_train]
    SX_test,  Sy_test  = X_test.loc[sample_test,:], y_test.loc[sample_test]
    
    train_sample = MyDataset(torch.FloatTensor(np.float64(np.array(SX_train))), 
                            torch.FloatTensor(np.float64(np.array(Sy_train))))
    test_sample  = MyDataset(torch.FloatTensor(np.float64(np.array(SX_test))), 
                           torch.FloatTensor(np.float64(np.array(Sy_test))))
    
    BATCH_SIZE = 150
    train_sample_loader = DataLoader(dataset=train_sample, batch_size=BATCH_SIZE, shuffle=True)
    test_sample_loader  = DataLoader(dataset=test_sample, batch_size=BATCH_SIZE, shuffle=False)
    
    # Training phase
    EPOCHS = 10
    LR = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    criterion = nn.MSELoss()
    optimizer_I = optim.Adamax(our_Imodel.parameters(), lr=LR)
    optimizer_D = optim.Adamax(our_Dmodel.parameters(), lr=LR)
    scheduler = None
    
    valid_loss, trial = train_model_pruning(
        trial, 
        our_Imodel, 
        our_Dmodel, 
        criterion,
        optimizer_I, 
        optimizer_D,
        train_sample_loader, 
        test_sample_loader, 
        scheduler, 
        device, 
        EPOCHS)
    
    return valid_loss
    

#%% Try it out
study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials=5, timeout=600)

pruned_trials = study.get_trials(
    deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(
    deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value)) 