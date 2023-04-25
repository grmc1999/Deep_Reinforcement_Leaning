from turtle import forward
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Neural_Net_module(nn.Module):
    def __init__(self,state_size,action_size,layer_sizes=[],activators=nn.ReLU()):
        super(Neural_Net_module,self).__init__()
        self.state_size=state_size
        self.action_size=action_size
        self.layer_sizes=[state_size]+layer_sizes+[action_size]
        self.activators=(activators if isinstance(activators,list) else [activators for _ in self.layer_sizes[:-1]])
        
        self.Modules=nn.ModuleList(
            [nn.Sequential(nn.Linear(inp,out),act) for inp,out,act in zip(self.layer_sizes[:-1],self.layer_sizes[1:],self.activators)]
        )
    def forward(self,state):
        for layer in self.Modules:
            state=layer(state)
        return state

class Neural_Net_REINFORCE_Actor(nn.Module):
    def __init__(self,state_size,action_size,layer_sizes=[],activators=nn.ReLU(),gamma=0.99):
        super(Neural_Net_REINFORCE_Actor,self).__init__()
        self.gamma=gamma
        self.state_size=state_size
        self.action_size=action_size
        self.layer_sizes=[state_size]+layer_sizes+[action_size]
        self.activators=(activators if isinstance(activators,list) else [activators for _ in self.layer_sizes[:-1]])

        self.Modules=nn.ModuleList(
            [nn.Sequential(nn.Linear(inp,out),act) for inp,out,act in zip(self.layer_sizes[:-1],self.layer_sizes[1:],self.activators)]
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
    


class Neural_Net_Actor_Critic(nn.Module):
    #def __init__(self,state_size,action_size,layer_sizes=[],activators=nn.ReLU(),gamma=0.99):
    def __init__(self,Actor_model,Critic_model,gamma=0.99,norm=(lambda x:x**2)):
        super(Neural_Net_Actor_Critic,self).__init__()
        self.gamma=gamma
        self.losses={"Actor_loss":0,
                     "Critic_loss":0}
        self.Actor=Actor_model
        self.Critic=Critic_model

        self.norm=(lambda x:x**2)

    def act(self,state):
        return self.Actor.forward(state)
    
    def cri(self,state):
        return self.Critic.forward(state)

    def compute_delta(self,R,gamma,s,s_p,done): #Consider as a constant
        if done:
            return R-self.cri(s)
        else:
            return R+gamma*self.cri(s_p).detach()-self.cri(s)

    def Actor_loss(self,cumulate_gama,delta,states,sampled_actions):

        actions=self.Actor.forward(states)
        logprobs=torch.log(actions)
        selected_logprobs=logprobs[np.arange(actions.shape[0]),sampled_actions]
        losses=cumulate_gama*delta*selected_logprobs
        return -losses.sum()
    
    def Critic_loss(self,delta,states):

        delta=self.norm(delta)
        #losses=delta*self.Critic.forward(states)
        losses=delta
        return losses.sum()
        
        

