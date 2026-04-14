import  torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import cma
from stable_baselines3 import PPO
from gymnasium import spaces


from vae import VAE,vae_loss
from mdrnn import LSTM,MDN,MDRNN,mdrnn_loss
from icm import ICM,icm_loss
from Controller import Controller


def collect_data(num_episodes=200):
    env=gym.make("LunarLander-v3")
    episodes=[]
    for ep in range(num_episodes):
        obs,_=env.reset()
        episode_data={
            "observations":[],
            "actions":[],
            "rewards":[],
            "next_observations":[],
            "dones":[],
        }
        done=False
        while not done:
            action=env.action_space.sample()
            obs_next,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated

            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["next_observations"].append(obs_next)
            episode_data["dones"].append(done)

            obs=obs_next

        episode_data["observations"]=torch.tensor(np.array(episode_data["observations"]),dtype=torch.float32)
        episode_data["actions"]=torch.tensor(episode_data["actions"],dtype=torch.long)
        episode_data["rewards"]=torch.tensor(episode_data["rewards"],dtype=torch.float32)
        episode_data["next_observations"]=torch.tensor(np.array(episode_data["next_observations"]),dtype=torch.float32)
        episode_data["dones"] = torch.tensor(episode_data["dones"], dtype=torch.bool)
        episodes.append(episode_data)

        if (ep+1)%50==0:
            print(f"Collected{ep+1}/{num_episodes}episodes")


    env.close()
    return episodes

def train_vae(episodes,num_epochs=200,lr=1e-3):
    vae=VAE(input_dim=8,z_dim=32)
    optimizer=torch.optim.Adam(vae.parameters(),lr=lr)

    all_obs=torch.cat([ep["observations"] for ep in episodes],dim=0)        #This is where all observation in combined
    print(f"Training VAE on {all_obs.shape[0]} observations")

    for epoch in range(num_epochs):
        beta=min(0.5,epoch/100)
        idx=torch.randperm(all_obs.shape[0])     #random permutation of indices
        all_obs_shuffled=all_obs[idx]           # Shuffling the data /per epoch
        epoch_recon=0
        epoch_kl=0
        num_batches=0

        batch_size=128                           #Mini Batch Training
        for i in range(0,all_obs.shape[0],batch_size): #instead of feeding 20,000 once we feed them in batches of 128
            batch=all_obs_shuffled[i:i+batch_size]
            x_hat,mu,log_var=vae(batch)
            total,recon,kl=vae_loss(batch,x_hat,mu,log_var,beta)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_recon+=recon.item()
            epoch_kl+=kl.item()
            num_batches+=1

        if(epoch+1)%20==0:
            print(f"Epoch{epoch+1}/{num_epochs}|Recon:{epoch_recon/num_batches:.4f}|KL:{epoch_kl/num_batches:.4f}|Beta:{beta:.2f}")

    return vae

def encode_episodes(vae,episodes):
    vae.eval()
    encoded_ep=[]
    with torch.no_grad():
        for ep in episodes:
            mu,log_var=vae.encode(ep["observations"])
            z=mu                                            #using mean, no sampling here
            encoded_ep.append({
                "z":z,
                "actions":ep["actions"],
                "rewards":ep["rewards"],
                "next_z":None                               #Later it'll be filled
            })
            mu_next,_=vae.encode(ep["next_observations"])
            encoded_ep[-1]["next_z"]=mu_next

    vae.train()
    return encoded_ep

def train_mdrnn(encoded_ep,num_epochs=200,lr=1e-3):
    z_dim=32
    action_dim=4
    hidden_dim=256
    num_mixtures=5

    mdrnn=MDRNN(z_dim,action_dim,hidden_dim,num_mixtures)
    optimizer=torch.optim.Adam(mdrnn.parameters(),lr=lr)
    print(f"Training MDRNN on {len(encoded_ep)} observations")
    for epoch in range(num_epochs):
        epoch_loss=0
        num_batches=0
        for ep in encoded_ep:
            z=ep["z"]
            actions=ep["actions"]
            if len(z)<2:
                continue

            z_seq=z[:-1].unsqueeze(0)
            a_one_hot=F.one_hot(actions[:-1],num_classes=action_dim).float()
            a_seq=a_one_hot.unsqueeze(0)
            z_target=z[1:].unsqueeze(0)

            pis,mus,sigmas=mdrnn(z_seq,a_seq)
            loss=mdrnn_loss(pis,mus,sigmas,z_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdrnn.parameters(),1.0)
            optimizer.step()
            epoch_loss+=loss.item()
            num_batches+=1

        if(epoch+1)%10==0:
            print(f"Epoch{epoch+1}/{num_epochs}|mdrnn Loss:{epoch_loss/num_batches}")


    return mdrnn


def train_icm(encoded_ep,num_epochs=200,lr=1e-3):
    icm=ICM(z_dim=32,action_dim=4,feature_dim=32)
    optimizer=torch.optim.Adam(icm.parameters(),lr=lr)

    all_z=[]                #Collecting all transitions
    all_actions=[]
    all_z_next=[]

    for ep in encoded_ep:
        z=ep["z"]
        actions=ep["actions"]
        z_next=ep["next_z"]

        a_one_hot=F.one_hot(actions,num_classes=4).float()
        all_z.append(z)
        all_actions.append(a_one_hot)
        all_z_next.append(z_next)

    all_z=torch.cat(all_z,dim=0)
    all_actions=torch.cat(all_actions,dim=0)
    all_z_next=torch.cat(all_z_next,dim=0)
    print(f"Training ICM on {all_z.shape[0]}transitions")

    for epoch in range(num_epochs):
        idx=torch.randperm(all_z.shape[0])
        batch_size=128
        epoch_fwd=0
        epoch_inv=0
        num_batches=0

        for i in range(0,all_z.shape[0],batch_size):
            b=idx[i:i+batch_size]
            z_t=all_z[b]
            a_t=all_actions[b]
            z_next=all_z_next[b]

            fwd_loss,inv_loss,r_int=icm(z_t,a_t,z_next)
            loss=0.8*inv_loss+0.2*fwd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_fwd+=fwd_loss.item()
            epoch_inv+=inv_loss.item()
            num_batches+=1

        if(epoch+1)%10==0:
            print(f"Epoch{epoch+1}/{num_epochs}|FWD:{epoch_fwd/num_batches:.4f}|INV:{epoch_inv/num_batches:.4f}")



    return icm


class WorldModel(gym.Env):
    def __init__(self,vae,mdrnn,icm_model,eta=0.01):
        super(WorldModel,self).__init__()

        self.vae=vae
        self.mdrnn=mdrnn
        self.icm_model=icm_model
        self.eta=eta

        self.env=gym.make("LunarLander-v3")                     #Real Env inside
        self.observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(288,),
            dtype=np.float32
        )
        self.action_space=spaces.Discrete(4)

        self.h=None
        self.c=None
        self.prev_action=None
        self.z_t=None

    def reset(self,seed=None,options=None):
        obs,info=self.env.reset(seed=seed)
        self.h=torch.zeros(1,256)
        self.c=torch.zeros(1,256)
        self.prev_action=torch.zeros(1,4)

        with torch.no_grad():
            obs_tensor=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
            mu,_=self.vae.encode(obs_tensor)
            self.z_t=mu
        combined=torch.cat([self.z_t,self.h],dim=-1).squeeze(0).numpy()
        return combined,info

    def step(self,action):
        with torch.no_grad():                       #MDN memory update
            a_one_hot=F.one_hot(torch.tensor([int(action)]),num_classes=4).float()
            x_t=torch.cat([self.z_t,self.prev_action],dim=-1)
            self.h,self.c=self.mdrnn.lstm(x_t,self.h,self.c)


        obs_next,reward,terminated,truncated,info=self.env.step(action)
        done=terminated or truncated

        with torch.no_grad():
            obs_tensor=torch.tensor(obs_next,dtype=torch.float32).unsqueeze(0) #Encode next observations
            mu_next,_=self.vae.encode(obs_tensor)
            z_next=mu_next
            _,_,r_int=self.icm_model(self.z_t,a_one_hot,z_next)
            total_reward=reward+self.eta*r_int.item()

            self.z_t=z_next
            self.prev_action=a_one_hot

        combined=torch.cat([self.z_t,self.h],dim=-1).squeeze(0).numpy()
        return combined,total_reward,terminated,truncated,info


    def close(self):
        self.env.close()



def train_controller(vae,mdrnn,icm_model,total_timesteps=100000,eta=0.01):
    print(f"Training Controller |{total_timesteps} timesteps| eta={eta}")
    env=WorldModel(vae,mdrnn,icm_model,eta=eta)
    model=PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    env.close()

    return model



def final_eval(model,vae,mdrnn,icm_model,num_episodes=10):
    env=WorldModel(vae,mdrnn,icm_model,eta=0)
    rewards=[]

    for episode in range(num_episodes):
        obs,_=env.reset()
        total_reward=0
        done=False
        while not done:
            action,_=model.predict(obs,deterministic=True)
            obs,reward,terminated,truncated,info=env.step(action)
            done=terminated or truncated
            total_reward+=reward

        rewards.append(total_reward)
        print(f"Episode:{episode+1}|Total Reward:{total_reward:.2f}")

    env.close()
    print(f"\nMean Reward:{np.mean(rewards):.2f}")
    print(f"\nBest Reward:{max(rewards):.2f}")
    print(f"\nWorst Reward:{min(rewards):.2f}")



def visualize(model,vae,mdrnn,icm_model,num_episodes=5):
    env=gym.make("LunarLander-v3",render_mode='human')
    wm_env=WorldModel(vae,mdrnn,icm_model,eta=0.01)
    wm_env.env=env                   #Replaced internal environment with the rendered one

    for episode in range(num_episodes):
        obs,_=wm_env.reset()
        total_reward=0
        done=False

        while not done:
            action,_=model.predict(obs,deterministic=True)
            obs,reward,terminated,truncated,info=wm_env.step(action)
            done=terminated or truncated
            total_reward+=reward


        print(f"Episode:{episode+1}|Total Reward:{total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    episodes = collect_data(num_episodes=200)
    vae = train_vae(episodes, num_epochs=200)
    encoded = encode_episodes(vae, episodes)
    mdrnn = train_mdrnn(encoded, num_epochs=150)
    icm_model = train_icm(encoded, num_epochs=100)
    model = train_controller(vae, mdrnn, icm_model, total_timesteps=200000, eta=0.0001)
    
    model.save("controller_ppo")
    torch.save(vae.state_dict(), "vae.pth")
    torch.save(mdrnn.state_dict(), "mdrnn.pth")
    torch.save(icm_model.state_dict(), "icm.pth")
    print("Models saved!")
    
    print("\n=== Final Evaluation ===")
    final_eval(model, vae, mdrnn, icm_model, num_episodes=10)
    
    print("\n=== Watching Agent Play ===")
    visualize(model, vae, mdrnn, icm_model, num_episodes=5)
   
