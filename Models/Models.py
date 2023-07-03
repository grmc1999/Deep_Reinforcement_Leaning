import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


class Neural_Net_module(nn.Module):
    def __init__(self,state_size,action_size,td,layer_sizes=[],activators=nn.ReLU(),batch_norm=None,dropouts=1):
        super(Neural_Net_module,self).__init__()
        self.state_size=state_size
        self.action_size=action_size
        self.layer_sizes=[state_size]+layer_sizes+[action_size]
        self.activators=(activators if isinstance(activators,list) else [activators for _ in self.layer_sizes[:-1]])
        self.batch_norm=(batch_norm if isinstance(batch_norm,list) else [batch_norm for _ in self.layer_sizes[:-1]])
        self.dropouts=(dropouts if isinstance(dropouts,list) else [dropouts for _ in self.layer_sizes[:-1]])
        self.td=td

        self.Modules=nn.ModuleList(
            [
                nn.Sequential(
            *([nn.Linear(inp,out)] +\
            [nn.BatchNorm1d(out)] if bn!=None else [] +\
            [act] +\
            [nn.Dropout(dpo)] if dpo!=None else [])
            ) for inp,out,bn,act,dpo in zip(
            self.layer_sizes[:-1],
            self.layer_sizes[1:],
            self.batch_norm,
            self.activators,
            self.dropouts
            )
            ]
        )
    def forward(self,state):
        for layer in self.Modules:
            state=layer(state)
        return state
    
class Koopman_neural_operator(nn.Module):
    def __init__(self,Encoder,Decoder):
        super(Koopman_neural_operator,self).__init__()

        self.Encoder=Encoder
        self.Decoder=Decoder
    
    def Encode(self,state):
        return self.Encoder(state)

    def Decode(self,ic):
        return self.Decoder(ic)

    #def get_K(self,ic):
    
    def forward(self,state):
        ic=self.Encoder(state)
        #omega,lambda=NN(ic) #2N omegas 2N lambdas
        #K=get_K(ic) #2Nx2N matrix
        #ic_=ic x K

        self.Decoder(ic)