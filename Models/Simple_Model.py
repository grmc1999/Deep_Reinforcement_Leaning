from turtle import forward
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Neural_Net_Actor(nn.Module):
    def __init__(self,state_size,action_size,layer_sizes=[],activators=nn.ReLU(),gamma=0.99):
        super(Neural_Net_Actor,self).__init__()
        self.gamma=gamma
        self.state_size=state_size
        self.action_size=action_size
        self.layer_sizes=[state_size]+layer_sizes+[action_size]
        self.activators=(activators if isinstance(activators,list) else [activators for _ in self.layer_sizes[:-1]])

        self.Modules=nn.ModuleList(
            [nn.Sequential([nn.Linear(inp,out),act]) for inp,out,act in zip(self.layer_sizes[:-1],self.layer_sizes[1:],self.activators)]
        )

        self.losses={"loss":0}
    
    def forward(self,state):
        for layer in self.Modules:
            state=layer(state)
        return state

    def act(self,state):
        return self.forward(state)

    def REINFORCE_loss(self,returns,states,sampled_actions):
        """
        returns: lambda return per steps in batched episodes [ steps_in_episode*episodes*batch_size ]
        states: states per steps in batched episodes [ steps_in_episode*episodes*batch_size ]
        OUTPUTS:
            Losess: Losses per episode
        """
        actions=self.forward(states) # [ steps_in_episode*episodes*batch_size, action_size ]
        logprobs=torch.log(actions)
        selected_logprobs=logprobs[np.arange(actions.shape[0]),sampled_actions]
        losses=-returns*selected_logprobs
        #losses=((returns.detach())*logprobs[np.arange(len(sampled_actions)),sampled_actions])

        return losses #
        
        

