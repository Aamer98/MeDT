'''
This script configures and executes experiments for evaluating recurrent autoencoding approaches useful for learning
informative representations of sequentially observed patient health.
After configuring the specific settings and hyperparameters for the selected autoencoder, the experiment can be specified to:
(1) Train the selected encoding and decoding functions used to establish the learned state representations 
(2) Evaluate the trained model and encode+save the patient trajectories by their learned representations
(3) Learn a treatment policy using the saved patient representations via offline RL. The algorithm used to learn a policy
    is the discretized form of Batch Constrained Q-learning [Fujimoto, et al (2019)]
The patient cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;
November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
'''

import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


################################################################################################################
# Prepare patient trajectories 
#
#
################################################################################################################
class Prepare_mimic():
    def __init__(self, train_data_file, validation_data_file, test_data_file, minibatch_size,
                 context_dim=0,  state_dim=40, num_actions=25):
        '''
        We assume discrete actions and scalar rewards!
        '''

        self.device = torch.device('cpu')
        self.train_data_file = train_data_file
        self.validation_data_file = validation_data_file
        self.test_data_file = test_data_file
        self.minibatch_size = minibatch_size

        self.num_actions = num_actions
        self.state_dim = state_dim

        self.context_dim = context_dim # Check to see if we'll remove the context from the input and only use it for decoding

        self.train_demog, self.train_states, self.train_interventions, self.train_lengths, self.train_times, self.acuities, self.rewards = torch.load(self.train_data_file)
        train_idx = torch.arange(self.train_demog.shape[0])
        self.train_dataset = TensorDataset(self.train_demog, self.train_states, self.train_interventions,self.train_lengths,self.train_times, self.acuities, self.rewards, train_idx)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.minibatch_size, shuffle=True)

        self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards = torch.load(self.validation_data_file)
        val_idx = torch.arange(self.val_demog.shape[0])
        self.val_dataset = TensorDataset(self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards, val_idx)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.minibatch_size, shuffle=False)

        self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_acuities, self.test_rewards = torch.load(self.test_data_file)
        test_idx = torch.arange(self.test_demog.shape[0])
        self.test_dataset = TensorDataset(self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_acuities, self.test_rewards, test_idx)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.minibatch_size, shuffle=False)

    def create_dataset(self):

        train_trajectories = []
        test_trajectories = []
        eval_trajectories = []

        ## LOOP THROUGH THE DATA
        with torch.no_grad():
            for i_set, loader in enumerate([self.train_loader, self.val_loader, self.test_loader]):

                for dem, obs, acu, _, timesteps, scores, rewards, _ in loader:
                    dem = dem.to(self.device)
                    obs = obs.to(self.device)
                    acu = acu.to(self.device)
                    scores = scores.to(self.device)
                    rewards = rewards.to(self.device)

                    max_length = int(timesteps.max().item())

                    obs = obs[:,:max_length,:]
                    dem = dem[:,:max_length,:]
                    acu = acu[:,:max_length,:]
                    scores = scores[:,:max_length,:]
                    rewards = rewards[:,:max_length]

                    # Loop over all transitions
                    trajectories = {'observations':[], 'dem_observations':[], 'actions':[], 'rewards':[],  'acuities':[]}
                    for i_trans in range(obs.shape[0]):
                        trajectories['observations'].append(obs[i_trans].numpy())
                        trajectories['dem_observations'].append(torch.cat((obs[i_trans], dem[i_trans]), dim=-1).numpy())
                        trajectories['actions'].append(acu[i_trans].argmax(dim=-1))
                        trajectories['rewards'].append(rewards[i_trans].numpy())
                        trajectories['acuities'].append(scores[i_trans].numpy())

                    trajectories['observations'] = np.array(trajectories['observations']).astype('float32').squeeze(0)
                    trajectories['dem_observations'] = np.array(trajectories['dem_observations']).astype('float32').squeeze(0)
                    trajectories['actions'] = trajectories['actions'][0].numpy().astype('int')
                    trajectories['rewards'] = np.array(trajectories['rewards']).astype('float32').squeeze(0)
                    trajectories['acuities'] = np.array(trajectories['acuities']).astype('float32').squeeze(0)

                    if i_set == 0:
                        train_trajectories.append(trajectories)
                    elif i_set == 1:
                        eval_trajectories.append(trajectories)
                    elif i_set == 2:
                        test_trajectories.append(trajectories)

            
            
            return train_trajectories, eval_trajectories, test_trajectories