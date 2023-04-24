import re
from tqdm import tqdm
import torch
import os
import numpy as np
from einops import repeat
from torch.distributions import Categorical


# View Returns calculation as a weigthing of losses, the return computation is implemented per training framework

class Episodic_learning(object):
    def __init__(self,model,free_input,env,Actor_optimizer_params,Critic_optimizer_params,res_dir,max_steps=200,cuda=False):
        
        self.max_steps=max_steps
        self.episodes_states={}
        self.episodes_action={}
        self.episodes_rewards={} #{1: [r1 r2 r3 ... ]]}

        self.episodes_states[0]=[]
        self.episodes_action[0]=[]
        self.episodes_rewards[0]=[]

        self.episodes_returns={}
        self.episodes_losses={0:"loss"} #{1: {"loss1":,"loss2":}}
        self.ep_limit=max_steps
        self.free_input=free_input
        self.env=env

        self.Ac_optimizer_params=Actor_optimizer_params
        self.Cr_optimizer_params=Critic_optimizer_params

        self.cuda=cuda
        self.model=model
        self.gamma=model.gamma
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
        
        s_p, reward, done, _=self.env.step(a.item())
        s_p=torch.from_numpy(s_p).float().unsqueeze(0)
        self.episodes_action[self.current_episode].append(a.item())
        self.episodes_rewards[self.current_episode].append(reward)
        
        if done:
            self.episodes_states[self.current_episode+1]=[]
            self.episodes_action[self.current_episode+1]=[]
            self.episodes_rewards[self.current_episode+1]=[]#consider size of rewards equal to 1 less than action and states
            s=self.env.reset()
            s=torch.from_numpy(s).float().unsqueeze(0) #[1,states]
        

        return s,s_p,reward,a,done            

    def Train(self,train_episodes):
        self.Ac_optim = torch.optim.Adam(self.model.Actor.Modules.parameters(), **(self.Ac_optimizer_params))
        self.Cr_optim = torch.optim.Adam(self.model.Critic.Modules.parameters(), **(self.Cr_optimizer_params))
        

        #for batch in tqdm(range(train_batches)):
        
        for episode in tqdm(range(train_episodes)):
            s=self.env.reset()
            s=torch.from_numpy(s).float().unsqueeze(0) #[1,states]
            Cum_gamma=1
            for step in tqdm(range(self.max_steps)):
                self.Ac_optim.zero_grad()
                self.Cr_optim.zero_grad()

                s,s_p,reward,action,done=self.run_episode_step(s)

                delta=self.model.compute_delta(reward,self.gamma,s,s_p,done)

                Act_loss=self.model.Actor_loss(
                    cumulate_gama=Cum_gamma,
                    delta=delta,
                    states=s,
                    sampled_actions=action
                )

                Cri_loss=self.model.Critic_loss(
                    delta=delta,
                    states=s,
                    norm=(lambda x:x**2)
                )

                Act_loss.backward()
                self.Ac_optim.step()

                Cri_loss.backward()
                self.Cr_optim.step()

                self.episodes_losses[self.current_episode]={step:{
                    "Actor_loss":Act_loss.detach().cpu().item(),
                    "Critic_loss":Cri_loss.detach().cpu().item()
                }
                    }
                Cum_gamma=Cum_gamma*self.gamma

                # TODO: if done episode
                if done:
                    self.current_episode=self.current_episode+1
                    break


            msg= ("\n").join(
                #[k+" {l:.8f}".format(l=(np.mean(np.vectorize(lambda x,loss: x[loss] )(np.array(self.episodes_losses[self.current_batch-1][k]),k)))) for k in self.model.losses.keys()] \
                [k+" {l:.8f}".format(l=(self.episodes_losses[self.current_episode-1][step][k])) for k in self.model.losses.keys()] \
                + ["Rewards mean {rm:.8f} Rewards std {rstd:.8f} Rewards sum {rs:.8f}".format(
                               rm=np.mean(np.array(self.episodes_rewards[self.current_episode-1])),
                               rstd=np.std(np.array(self.episodes_rewards[self.current_episode-1])),
                               rs=np.sum(np.array(self.episodes_rewards[self.current_episode-1]))
                               )])
            tqdm.write(
              msg
          )
          #save history

            np.save(os.path.join(self.res_dir,' STATES.npy'), self.episodes_states)
            np.save(os.path.join(self.res_dir,' ACTIONS.npy'), self.episodes_action)
            np.save(os.path.join(self.res_dir,' REWARDS.npy'), self.episodes_rewards)
            np.save(os.path.join(self.res_dir,' LOSSES.npy'), self.episodes_losses) 


          
        

    