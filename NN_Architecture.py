import torch
import torch.nn as nn

#%% INVERSE: from frequencies to parameters. Input blocks + FC.
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
        self.APPLY_BN = True
    
    def build_LinearBlock(self):
        """ Builds a single input block (there is one per band). """
        layers = []
        in_ = self.D_IN
        for D_H in self.D_HIDDEN_BK:
            layers.append(nn.Linear(in_, D_H))
            layers.append(nn.ReLu())
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
            layers.append(nn.ReLu())
            if self.APPLY_DROPOUT: 
                layers.append(nn.Dropout(self.P_DROPOUT))
            if self.APPLY_BN:
                layers.append(nn.BatchNorm1d(D_H))
            in_ = D_H
            
        layers.append(nn.Linear(in_, self.D_OUT))
            
        return nn.Sequential(*layers)
    
    
    def forward_InverseSplitData(self, DATA):
        """ 
        Apply blocks to the input data:
        @param DATA: input data within a batch (BATCH_SIZE x D_IN)
        """
        fbands = torch.split(DATA, self.N_BLOCKS, dim=1)  # Frequency bands = columns
        assert(max(fbands[0].size()) == self.D_IN)
        
        outputs = []
        for fband in fbands:
            out_ = self.build_LinearBlock(fband)
            out_ = out_.view(out_.size(0), -1)
            outputs.append(out_)
        
        return torch.cat(outputs, dim=1)
    
    
    def forward(self, DATA):
        """ All the NN: Blocks + FC """
        out_ = []
        for d in DATA:
            out_.append(self.forward_InverseSplitData(d))
        x = torch.tensor(out_)
        x = self.build_FC(x)
        
        return x  # Parameters


#%% DIRECT: from parameters to frequencies. FC only.
class DirectNN(nn.Module):
    def __init__(self,
                 D_IN: int, 
                 D_HIDDEN: list,
                 D_OUT: int, 
                 P_DROPOUT: None):
        """
        Inverse NN, i.e. from parameters to frequencies:
        @param D_IN: number of inputs for the FC
        @param D_HIDDEN: list with the dimensions of each hidden layer in the FC
        @param D_OUT: dimension of the output layer in the FC, i.e. band frequencies
        @param P_DROPOUT: dropout probability
        """
        super(InverseNN, self).__init__()
        self.D_IN      = D_IN
        self.D_HIDDEN  = D_HIDDEN
        self.D_OUT     = D_OUT
        self.P_DROPOUT = P_DROPOUT
        
        # Apply dropout?
        self.APPLY_DROPOUT = True
        if self.P_DROPOUT == None:
            self.APPLY_DROPOUT = False
            
        # Apply Batch Normalization?
        self.APPLY_BN = True
        
    
    def build_FC(self):
        layers = []
        in_ = self.D_IN
        for D_H in self.D_HIDDEN:
            layers.append(nn.Linear(in_, D_H))
            layers.append(nn.ReLu())
            if self.APPLY_DROPOUT:
                layers.append(nn.Dropout(self.P_DROPOUT))
            if self.APPLY_BN:
                layers.append(nn.BatchNorm1d(D_H))
            in_ = D_H
            
        layers.append(nn.Linear(in_, self.D_OUT))
        
        return nn.Sequential(*layers)
    
    
    def forward(self, DATA):
        """ All the NN: Blocks + FC """
        x = self.build_FC(DATA)
        
        return x  # Frequency


# - Construct residual NN?