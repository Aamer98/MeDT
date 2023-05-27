from torch.utils.data import Dataset, DataLoader
import torch

import pickle
import random
import numpy as np


################################################################################################################
# Dataset
#
#
################################################################################################################
class MimicTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale, norm=False):

        self.context_len = context_len        

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['dem_observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['dem_observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        if norm==True:
            for traj in self.trajectories:
                traj['dem_observations'] = (traj['dem_observations'] - self.state_mean) / self.state_std

        for i in range(len(self.trajectories)):
            self.trajectories[i]['acuities'] = np.concatenate((self.trajectories[i]['acuities'][1:,:], np.reshape((self.trajectories[i]['acuities'][-1,:]), (1,10))), axis=0) #reforumulate as acuity score of next time step


    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['dem_observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['dem_observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            saps = torch.from_numpy(traj['acuities'][si : si + self.context_len,2])

            div_saps = torch.from_numpy(traj['acuities'][si : si + self.context_len,3:])

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

            

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['dem_observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)], 
                               dim=0)
            
            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)], 
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)], 
                               dim=0)
            
            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long), 
                                   torch.zeros(padding_len, dtype=torch.long)], 
                                  dim=0)
            
            saps = torch.from_numpy(traj['acuities'][:,2])
            saps = torch.cat([saps,
                                torch.zeros(([padding_len] + list(saps.shape[1:])),
                                dtype=saps.dtype)], 
                               dim=0)
            
            div_saps = torch.from_numpy(traj['acuities'][:,3:])
            div_saps = torch.cat([div_saps,
                                torch.zeros(([padding_len] + list(div_saps.shape[1:])),
                                dtype=div_saps.dtype)], 
                               dim=0)            


        return  states, actions, returns_to_go, timesteps, saps, div_saps, traj_len


################################################################################################################
# Discounting
#
#
################################################################################################################
def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum