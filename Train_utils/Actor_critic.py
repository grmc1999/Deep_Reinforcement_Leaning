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
    def __init__(self,model,free_input,env,Actor_optimizer_params,Critic_optimizer_params,res_dir,phi,multi_opt=True,max_steps=200,cuda=False,sch_f=lambda x:x):
        
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

        if self.multi_opt:
            self.Ac_optim = getattr(torch.optim,self.Ac_optimizer_params["name"])(self.model.Actor.Modules.parameters(), **(self.Ac_optimizer_params["args"]))
            self.Cr_optim = getattr(torch.optim,self.Cr_optimizer_params["name"])(self.model.Critic.Modules.parameters(), **(self.Cr_optimizer_params["args"]))
        else:
            self.Ac_optim = getattr(torch.optim,self.Ac_optimizer_params["name"])(list(self.model.Actor.Modules.parameters())+list(self.model.Critic.Modules.parameters()), **(self.Ac_optimizer_params["args"]))

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

        return s,s_p,reward,pa,a,done            

    def Train(self,train_episodes,T,phi,static=True,modified_reward=False):

        
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
                

                s,s_p,reward,p_actions,action,done=self.run_episode_step(s)

                if modified_reward:
                    reward=reward*step

                if step==(self.max_steps-1):
                    done=True

                delta=self.model.compute_delta(reward,self.gamma,s,s_p,done)

                Act_loss=self.model.Actor_loss(
                    cumulate_gama=Cum_gamma,
                    delta=delta.detach(),
                    states=s,
                    prob_actions=p_actions,
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
                s=s_p

                # TODO: if done episode
                if done or step==(self.max_steps-1):
                    self.episodes_states[self.current_episode+1]=[]
                    self.episodes_action[self.current_episode+1]=[]
                    self.episodes_rewards[self.current_episode+1]=[]#consider size of rewards equal to 1 less than action and states
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


          
        

class n_step_learning(Episodic_learning):
    def __init__(self,model,free_input,n_steps,env,Actor_optimizer_params,Critic_optimizer_params,res_dir,phi,act_loss_type="Actor_loss_TD",multi_opt=True,max_steps=200,cuda=False,sch_f=lambda x:x):
        super().__init__(model,free_input,env,Actor_optimizer_params,Critic_optimizer_params,res_dir,phi,multi_opt,max_steps,cuda,sch_f)

        self.n_steps=n_steps
        self.act_loss_type=act_loss_type
    
    def run_episode_n_steps(self,s,n):
        # Run N steps and use error over V_t
        S=[]
        R=[]
        pA=[]
        A=[]
        d=[]

        for i in range(n):
            self.episodes_states[self.current_episode].append(s)
            #save S
            S.append(s)
            if self.cuda:
                s=s.cuda()
            pa=self.model.act(s)
            sampler=Categorical(pa)
            a=sampler.sample().detach()

            s_p, reward, done, _,_=self.env.step(a.item())
            s_p=torch.from_numpy(s_p).float().unsqueeze(0)
            s=s_p

            self.episodes_action[self.current_episode].append(a.item())
            self.episodes_rewards[self.current_episode].append(reward)

            R.append(torch.tensor([[reward]]))
            pA.append(pa)
            A.append(a.unsqueeze(-1))
            d.append(torch.tensor([[done]]))
            if done:
                break

        S.append(s)
        return torch.cat(S),torch.cat(R),torch.cat(pA),torch.cat(A),torch.cat(d)
    
    def Train(self,train_episodes,T,phi,static=True,modified_reward=False):

        
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
                

                S,R,pA,A,done=self.run_episode_n_steps(s,self.n_steps)

                if step==(self.max_steps-1):
                    done=True

                #TODO: Compute_n_delta
                delta,_=self.model.compute_n_delta(R,self.gamma,S,done)

                args=(Cum_gamma,S,pA,A,R,done)
                Act_loss=getattr(self.model,self.act_loss_type)(*(args))

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
                
                #TODO: Cummulate gamma considering steps
                Cum_gamma=Cum_gamma*(self.gamma**self.n_steps)
                s=S[-1]

                if done[-1,0]. or step==(self.max_steps-1):
                    self.episodes_states[self.current_episode+1]=[]
                    self.episodes_action[self.current_episode+1]=[]
                    self.episodes_rewards[self.current_episode+1]=[]#consider size of rewards equal to 1 less than action and states
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