from turtle import forward
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Neural_Net_Actor(nn.Module):
    def __init__(self,state_size,action_size,gamma):
        super(Neural_Net_Actor,self).__init__()
        self.gamma=gamma
        self.state_size=state_size
        self.action_size=action_size
        self.layer_1=nn.Linear(state_size,16)
        self.layer_2=nn.Linear(10,50)
        self.layer_3=nn.Linear(50,50)
        self.layer_4=nn.Linear(50,25)
        self.layer_5=nn.Linear(25,12)
        self.layer_6=nn.Linear(12,12)
        self.layer_7=nn.Linear(16,6)
        self.layer_8=nn.Linear(16,action_size)

        self.activator_1=nn.LeakyReLU()
        self.activator_5=nn.Softmax(dim=-1  )

        self.losses={"loss":0}
    
    def forward(self,state):
        state=self.activator_1(self.layer_1(state))
        #state=self.activator_1(self.layer_2(state))
        #state=self.activator_1(self.layer_3(state))
        #state=self.activator_1(self.layer_4(state))
        #state=self.activator_1(self.layer_5(state))
        #state=self.activator_1(self.layer_6(state))
        #state=self.activator_1(self.layer_7(state))
        state=self.activator_5(self.layer_8(state))
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
        print("\n actions")
        print(actions)
        logprobs=torch.log(actions)
        print("\n log probs")
        print(logprobs)
        #selected_logprobs=logprobs[np.arange(actions.shape[0]),sampled_actions]
        #losses=returns*selected_logprobs
        losses=-(returns*logprobs[np.arange(len(sampled_actions)),sampled_actions])
        print("\n losses")
        print(losses)

        return losses #
        
        

