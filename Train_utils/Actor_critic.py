import re
from tqdm import tqdm
import torch
import os
import numpy as np
#from einops import repeat
from torch.distributions import Categorical


# View Returns calculation as a weigthing of losses, the return computation is implemented per training framework
#lambda x:(1-2e-8)*np.cos(2*np.pi*x)+(0.5+1e-8)
class Episodic_learning(object):
    def __init__(self,model,free_input,env,Actor_optimizer_params,Critic_optimizer_params,res_dir,phi,multi_opt=True,max_steps=200,cuda=False,sch_f=lambda x:(1-2e-8)*np.cos(2*np.pi*x)+(0.5+1e-8)):
        
        self.sch_f=sch_f

        self.max_steps=max_steps
        self.episodes_states={}
        self.episodes_action={}
        self.episodes_rewards={} #{1: [r1 r2 r3 ... ]]}

        self.episodes_states[0]=[]
        self.episodes_action[0]=[]
        self.episodes_rewards[0]=[]

        self.episodes_returns={}
        self.episodes_losses={0:{}} #{1: {"loss1":,"loss2":}}
        self.ep_limit=max_steps
        self.free_input=free_input
        self.env=env

        self.multi_opt=multi_opt
        self.Ac_optimizer_params=Actor_optimizer_params
        self.Cr_optimizer_params=Critic_optimizer_params

        self.cuda=cuda
        self.model=model
        self.gamma=model.gamma
        self.phi=phi
        self.current_episode=0
        self.device=("cuda" if cuda else "cpu")
        self.res_dir=res_dir
        if cuda:
            self.model.cuda()

    def run_episode_step(self,s):
        u=self.free_input
        

        #Generate episode
        #for ep_step in tqdm(range(self.ep_limit+1)):
        self.episodes_states[self.current_episode].append(s)
        #s=torch.from_numpy(s).float().unsqueeze(0) #[1,states]
        if self.cuda:
            s=s.cuda()
        pa=self.model.act(s)
        sampler=Categorical(pa)
        a=sampler.sample().detach()
        
        s_p, reward, done, _,_=self.env.step(a.item())
        s_p=torch.from_numpy(s_p).float().unsqueeze(0)
        self.episodes_action[self.current_episode].append(a.item())
        self.episodes_rewards[self.current_episode].append(reward)
        
        if done:
            self.episodes_states[self.current_episode+1]=[]
            self.episodes_action[self.current_episode+1]=[]
            self.episodes_rewards[self.current_episode+1]=[]#consider size of rewards equal to 1 less than action and states
            #s=self.env.reset()[0]
            #s=torch.from_numpy(s).float().unsqueeze(0) #[1,states]
        

        return s,s_p,reward,a,done            

    def Train(self,train_episodes,T,phi,static=True,modified_reward=False):
        if self.multi_opt:
            self.Ac_optim = torch.optim.SGD(self.model.Actor.Modules.parameters(), **(self.Ac_optimizer_params))
            self.Cr_optim = torch.optim.SGD(self.model.Critic.Modules.parameters(), **(self.Cr_optimizer_params))
        else:
            self.Ac_optim = torch.optim.Adam(list(self.model.Actor.Modules.parameters())+list(self.model.Critic.Modules.parameters()), **(self.Ac_optimizer_params))
        
        for episode in tqdm(range(train_episodes)):
            s=self.env.reset()[0]
            s=torch.from_numpy(s).float().unsqueeze(0) #[1,states]
            Cum_gamma=1
            self.episodes_losses[self.current_episode]={0:{}}

            #MODIFY self.phi DINAMICALLY
            #self.phi=np.cos()
            if not static:
                self.phi=self.sch_f((episode%T)/T)
            else:
                self.phi=phi
            for step in tqdm(range(self.max_steps)):
                

                s,s_p,reward,action,done=self.run_episode_step(s)

                if modified_reward:
                    reward=reward*step

                delta=self.model.compute_delta(reward,self.gamma,s,s_p,done)

                Act_loss=self.model.Actor_loss(
                    cumulate_gama=Cum_gamma,
                    delta=delta.detach(),
                    states=s,
                    sampled_actions=action
                )

                Cri_loss=self.model.Critic_loss(
                    delta=delta,
                    states=s
                )

                
                

                if self.multi_opt:
                    self.Ac_optim.zero_grad()
                    self.Cr_optim.zero_grad()
                    Act_loss.backward()
                    self.Ac_optim.step()

                    Cri_loss.backward()
                    self.Cr_optim.step()
                else:
                    self.Ac_optim.zero_grad()
                    Total_loss=self.phi*Cri_loss+(1-self.phi)*Act_loss
                    Total_loss.backward()
                    self.Ac_optim.step()

                self.episodes_losses[self.current_episode].update({step:{
                    "Actor_loss":Act_loss.detach().cpu().item(),
                    "Critic_loss":Cri_loss.detach().cpu().item()
                }
                    })
                Cum_gamma=Cum_gamma*self.gamma

                # TODO: if done episode
                if done:
                    self.current_episode=self.current_episode+1
                    break


            msg= ("\n").join(
                [k+" {l:.8f}".format(l=( np.mean(np.array(list(map( lambda st: self.episodes_losses[self.current_episode-1][st][k],list(range(step)) )))) )) for k in self.model.losses.keys()] \
                + ["Rewards mean {rm:.8f} Rewards std {rstd:.8f} Rewards sum {rs:.8f}".format(
                               rm=np.mean(np.array(self.episodes_rewards[self.current_episode-1])),
                               rstd=np.std(np.array(self.episodes_rewards[self.current_episode-1])),
                               rs=np.sum(np.array(self.episodes_rewards[self.current_episode-1]))
                               )])
            tqdm.write(
              msg
          )
          #save history

        np.save(os.path.join(self.res_dir,'STATES.npy'), self.episodes_states)
        np.save(os.path.join(self.res_dir,'ACTIONS.npy'), self.episodes_action)
        np.save(os.path.join(self.res_dir,'REWARDS.npy'), self.episodes_rewards)
        np.save(os.path.join(self.res_dir,'LOSSES.npy'), self.episodes_losses) 


          
        

    