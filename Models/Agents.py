#from turtle import forward
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


class Neural_Net_module(nn.Module):
    def __init__(self,state_size,action_size,layer_sizes=[],activators=nn.ReLU(),dropouts=1):
        super(Neural_Net_module,self).__init__()
        self.state_size=state_size
        self.action_size=action_size
        self.layer_sizes=[state_size]+layer_sizes+[action_size]
        self.activators=(activators if isinstance(activators,list) else [activators for _ in self.layer_sizes[:-1]])
        self.dropouts=(dropouts if isinstance(dropouts,list) else [dropouts for _ in self.layer_sizes[:-1]])
        
        self.Modules=nn.ModuleList(
            [nn.Sequential(nn.Linear(inp,out),act,nn.Dropout(dpo) if dpo!=None else nn.Identity()) for inp,out,act,dpo in zip(self.layer_sizes[:-1],self.layer_sizes[1:],self.activators,self.dropouts)]
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

        self.norm=norm

    def act(self,state):
        return self.Actor.forward(state)
    
    def cri(self,state):
        return self.Critic.forward(state)

    def compute_delta(self,R,gamma,s,s_p,done): #Consider as a constant
        if done:
            return R-self.cri(s)
        else:
            return R+gamma*self.cri(s_p).detach()-self.cri(s)

    def Actor_loss(self,cumulate_gama,delta,states,prob_actions,sampled_actions):

        #prob_actions=self.Actor.forward(states)
        logprobs=torch.log(prob_actions)
        #selected_logprobs=logprobs[np.arange(prob_actions.shape[0]),sampled_actions]
        selected_logprobs=logprobs[torch.cat((
            torch.tensor(np.arange(prob_actions.shape[0]).reshape(-1,1)),
            sampled_actions.unsqueeze(0)
            ),dim=1).T.detach().numpy()] #[n,1]
        losses=cumulate_gama*delta*selected_logprobs
        return -losses.sum()
    
    def Critic_loss(self,delta,states):

        delta=self.norm(delta)
        #losses=delta*self.Critic.forward(states)
        losses=delta
        return losses.sum()

class Neural_Net_n_step_Actor_Critic(Neural_Net_Actor_Critic):
    def __init__(self,Actor_model,Critic_model,gamma=0.99,norm=(lambda x:x**2),entropy_w=0):
        super(Neural_Net_n_step_Actor_Critic,self).__init__(Actor_model,Critic_model,gamma,norm)
        self.entropy_w=entropy_w
    
    def compute_n_delta(self,R,gamma,S,done):
        if done[-1]:
            G=torch.sum((torch.tensor(gamma**np.arange(len(R)).reshape(-1,1)))*R)
        else:
            G=torch.sum((torch.tensor(gamma**np.arange(len(R)).reshape(-1,1)))*R) + (gamma**len(R))*(self.cri(S[-1])).detach()
        delta=G-self.cri(S[0])
        return delta,G
#TODO: Check cumulate gammas
    def Actor_loss_TD(self,cumulate_gama,S,pA,A,R,done):
        #TODO: modificar para entradas muliples
        logprobs=torch.log(pA) #[n,3]
        selected_logprobs=logprobs[torch.cat((torch.tensor(np.arange(pA.shape[0]).reshape(-1,1)),A),dim=1).T.detach().numpy()] #[n,1]
        cumulate_gama=cumulate_gama*((self.gamma)**torch.tensor(np.arange(pA.shape[0])))
        delta=torch.cat([(self.compute_delta(r,self.gamma,s,s_p,d)).detach() for r,s,s_p,d in zip(R,S[:-1],S[1:],done)])
        losses=cumulate_gama*delta*selected_logprobs
        if self.entropy_w>0:
            return -losses.sum()/len(R) - self.entropy_w*(torch.tensor(list(map(lambda p:Categorical(p).entropy(),pA))).mean())
        else:
            return -losses.sum()/len(R)

    def Actor_loss_G(self,cumulate_gama,S,pA,A,R,done):
        logprobs=torch.log(pA)
        selected_logprobs=logprobs[torch.cat((torch.tensor(np.arange(pA.shape[0]).reshape(-1,1)),A),dim=1).T.detach().numpy()] #[n,1]
        _,G=self.compute_n_delta(R,self.gamma,S,done)
        losses=cumulate_gama*G*selected_logprobs
        if self.entropy_w>0:
            return -losses.sum()/len(R) - self.entropy_w*(torch.tensor(list(map(lambda p:Categorical(p).entropy(),pA))).mean())
        else:
            return -losses.sum()/len(R)
    
    def Actor_loss_TTD(self,cumulate_gama,S,pA,A,R,done):

        #prob_actions=self.Actor.forward(states)
        logprobs=torch.log(pA)
        selected_logprobs=logprobs[torch.cat((torch.tensor(np.arange(pA.shape[0]).reshape(-1,1)),A),dim=1).T.detach().numpy()] #[n,1]
        delta,_=self.compute_n_delta(R,self.gamma,S,done)
        losses=cumulate_gama*(delta.detach())*selected_logprobs
        if self.entropy_w>0:
            return -losses.sum()/len(R) - self.entropy_w*(torch.tensor(list(map(lambda p:Categorical(p).entropy(),pA))).mean())
        else:
            return -losses.sum()/len(R)


    #TODO: Decide 
    #OP1: n TD deltas
    #OP2: G_t:t+n
        
class DINO_n_step_Actor_Critic(Neural_Net_Actor_Critic):
    def __init__(self,Actor_model,Critic_model,gamma=0.99,norm=(lambda x:x**2),entropy_w=0):
        super(Neural_Net_n_step_Actor_Critic,self).__init__(Actor_model,Critic_model,gamma,norm)
        self.entropy_w=entropy_w
        # MAKE COPY OF Actor_model -> student, teacher
        # DEL self.Critic

    
    def compute_n_delta(self,R,gamma,S,done):
        if done[-1]:
            G=torch.sum((torch.tensor(gamma**np.arange(len(R)).reshape(-1,1)))*R)
        else:
            G=torch.sum((torch.tensor(gamma**np.arange(len(R)).reshape(-1,1)))*R) + (gamma**len(R))*(self.cri(S[-1])).detach()
        delta=G-self.cri(S[0])
        return delta,G
    
    #def gen_history_episodes():
        
    #def SSSL(self,module,SL_loss): 3SL #TO interchange between critic and actor
        #Generate N views of history
        #for each view
            #critic forward
            #H loss
            #Regression loss on teacher or student
    #def H_loss
#TODO: Check cumulate gammas
    def Actor_loss_TD(self,cumulate_gama,S,pA,A,R,done):
        #TODO: modificar para entradas muliples
        logprobs=torch.log(pA) #[n,3]
        selected_logprobs=logprobs[torch.cat((torch.tensor(np.arange(pA.shape[0]).reshape(-1,1)),A),dim=1).T.detach().numpy()] #[n,1]
        cumulate_gama=cumulate_gama*((self.gamma)**torch.tensor(np.arange(pA.shape[0])))
        delta=torch.cat([(self.compute_delta(r,self.gamma,s,s_p,d)).detach() for r,s,s_p,d in zip(R,S[:-1],S[1:],done)])
        losses=cumulate_gama*delta*selected_logprobs
        if self.entropy_w>0:
            return -losses.sum()/len(R) - self.entropy_w*(torch.tensor(list(map(lambda p:Categorical(p).entropy(),pA))).mean())
        else:
            return -losses.sum()/len(R)

    def Actor_loss_G(self,cumulate_gama,S,pA,A,R,done):
        logprobs=torch.log(pA)
        selected_logprobs=logprobs[torch.cat((torch.tensor(np.arange(pA.shape[0]).reshape(-1,1)),A),dim=1).T.detach().numpy()] #[n,1]
        _,G=self.compute_n_delta(R,self.gamma,S,done)
        losses=cumulate_gama*G*selected_logprobs
        if self.entropy_w>0:
            return -losses.sum()/len(R) - self.entropy_w*(torch.tensor(list(map(lambda p:Categorical(p).entropy(),pA))).mean())
        else:
            return -losses.sum()/len(R)
    
    def Actor_loss_TTD(self,cumulate_gama,S,pA,A,R,done):

        #prob_actions=self.Actor.forward(states)
        logprobs=torch.log(pA)
        selected_logprobs=logprobs[torch.cat((torch.tensor(np.arange(pA.shape[0]).reshape(-1,1)),A),dim=1).T.detach().numpy()] #[n,1]
        delta,_=self.compute_n_delta(R,self.gamma,S,done)
        losses=cumulate_gama*(delta.detach())*selected_logprobs
        if self.entropy_w>0:
            return -losses.sum()/len(R) - self.entropy_w*(torch.tensor(list(map(lambda p:Categorical(p).entropy(),pA))).mean())
        else:
            return -losses.sum()/len(R)

