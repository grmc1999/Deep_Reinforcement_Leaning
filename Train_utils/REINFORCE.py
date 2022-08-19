import re
from tqdm import tqdm
import torch
import os
import numpy as np
from einops import repeat
from torch.distributions import Categorical


# View Returns calculation as a weigthing of losses, the return computation is implemented per training framework

class REINFORCE(object):
    def __init__(self,model,ep_limit,free_input,env,gamma,optimizer_params,res_dir,batch_size=1,cuda=False):
        self.episodes_states={}
        self.episodes_action={}
        self.episodes_rewards={} #{1: [r1 r2 r3 ... ]]}
        self.episodes_returns={}
        self.episodes_losses={0:"loss"} #{1: {"loss1":,"loss2":}}
        self.ep_limit=ep_limit
        self.free_input=free_input
        self.env=env
        self.optimizer_params=optimizer_params
        self.cuda=cuda
        self.model=model
        self.gamma=gamma
        self.batch_size=batch_size
        self.current_episode=0
        self.current_batch=0
        self.device=("cuda" if cuda else "cpu")
        self.res_dir=res_dir

    #def action(self):
    def compute_return(self,Rs,gamma_w):
        gamma_w=gamma_w*np.arange(1,Rs.shape[0]+1)
        return np.sum(gamma_w*Rs)

    def compute_episode_returns(self,Rs,gamma_w):
        tm=np.arange(0,Rs.shape[0])-np.arange(Rs.shape[0]).reshape(-1,1)
        zf=(tm>=0)
        tm=(gamma_w**tm)*zf
        Rs=np.tile(Rs,(Rs.shape[0],1))
        return (Rs*tm).sum(1)

    def run_episode(self):
        u=self.free_input
        s=self.env.reset()
        self.episodes_states[self.current_episode]=[s]
        self.episodes_action[self.current_episode]=[u]
        self.episodes_rewards[self.current_episode]=[0]#consider size of rewards equal to 1 less than action and states


        #Generate episode
        for ep_step in tqdm(range(self.ep_limit+1)):
            s=torch.from_numpy(s).float().unsqueeze(0) #[1,states]
            if self.cuda:
                s=s.cuda()
            pa=self.model.act(s)
            sampler=Categorical(pa)
            a=sampler.sample()

            s, reward, done, _=self.env.step(a.item())
            self.episodes_states[self.current_episode].append(s)
            self.episodes_action[self.current_episode].append(a.item())
            self.episodes_rewards[self.current_episode].append(reward)
            if done:
                break
        self.episodes_returns[self.current_episode]=self.compute_episode_returns(np.array(self.episodes_rewards[self.current_episode][:]),
                                    self.gamma)
        self.current_episode=self.current_episode+1

    def batch_episodes(self):
        for batch in tqdm(range(self.batch_size)):
            self.run_episode()

            
        episode_list=np.arange(self.batch_size)+self.batch_size*self.current_batch
        self.current_batch=self.current_batch+1
        states_batch=torch.from_numpy(np.concatenate(list(map(self.episodes_states.get,episode_list)))).float().to(self.device)
        action_batch=torch.from_numpy(np.concatenate(list(map(self.episodes_action.get,episode_list)))).long().to(self.device)
        rewards_batch=torch.from_numpy(np.concatenate(list(map(self.episodes_rewards.get,episode_list)))).float().to(self.device)
        returns_batch=torch.from_numpy(np.concatenate(list(map(self.episodes_returns.get,episode_list)))).float().to(self.device)
        return states_batch,action_batch,rewards_batch,returns_batch


            

    def Train(self,train_batches):
        self.optim = torch.optim.Adam(self.model.parameters(), **(self.optimizer_params))

        for batch in tqdm(range(train_batches)):
            states_batch,action_batch,rewards_batch,returns_batch=self.batch_episodes()

            losses=self.model.REINFORCE_loss(
            returns=returns_batch,
            states=states_batch,
            sampled_actions=action_batch
          )
            self.optim.zero_grad()
          #TODO: for generalization implement compute losses
            losses=-losses.mean()
            losses.backward()
            self.optim.step()
            self.episodes_losses[self.current_batch-1]={"loss":losses.cpu().item()}

            msg= ("\n").join(
                #[k+" {l:.8f}".format(l=(np.mean(np.vectorize(lambda x,loss: x[loss] )(np.array(self.episodes_losses[self.current_batch-1][k]),k)))) for k in self.model.losses.keys()] \
                [k+" {l:.8f}".format(l=(self.episodes_losses[self.current_batch-1][k])) for k in self.model.losses.keys()] \
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


          
        

    