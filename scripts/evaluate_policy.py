import os
import argparse

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np

from models.state_predictor import State_Predictor
from models.MeDT import MeDT
from models.GPT import GPTConfig
from datasets import mimic_dataset
from utils import set_seed, state_sample, action_sample, calculate_scores


################################################################################################################
# Constants
#
################################################################################################################
CONTEXT_LENGTH = 20
RTG_SCALE = 1
ACTION_VOCAB_SIZE = 25
STATE_VOCAB_SIZE = 45 
N_HEAD = 8 
WARMPUP_TOKENS = 512*20

device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
set_seed(0)


################################################################################################################
# test_loop
#
# This is where learning happens. More specifically, the policy network is trained to predict actions  
# with teacher forcing using cross-entropy loss.
#
#
################################################################################################################
def evaluate_model(policy_model, state_predictor, ret, data_loader):
    
    predicted_states = []
    
    for x, y, r, t, SAPS, SAPS_constituents, traj_len  in data_loader:
        
        x = x.to(device) # states
        y = y.to(device) # actions
        t = t.to(device) # timesteps

        # Pick initial state
        state = x[:,0,:].to(device).type(torch.float32).to(device).unsqueeze(0)  

        # ATG of first state
        atg = SAPS_constituents[:,0,:].type(torch.float32).unsqueeze(0) 
        # Vary ATG by specified %
        atg = atg + torch.ones_like(atg)*(args.perc/100) 
        # Remove negative values
        atg[atg < 0] = 0 
        atg = atg.to(device)

        atgs = atg.clone()
        rtgs = [ret]

        # Specify traj_len for PromptDTATG
        if args.model_type == 'PromptDTATG':
                traj_len = torch.ones_like(traj_len) 

        # First state is from env, first rtg is target return, and first timestep is 0
        sampled_action = action_sample(policy_model, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=t[:,:state.shape[1]].unsqueeze(-1), atgs=atgs, traj_len=traj_len) 

        j = 0
        episode_states = state.clone()
        actions = []

        # ATG of prompt
        if args.model_type != 'PromptDTATG':
                atg = atg + torch.ones_like(atg)*(args.perc/100)
                atg[atg < 0] = 0
                atgs = torch.cat([atgs, atg], dim=1)


        # Iterate over 10 time steps
        for i_step in range(10): 

                actions += [sampled_action]

                rtgs += [ret]

                state = state_sample(state_predictor, episode_states, 1, 
                        actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                        timesteps=t[:,:episode_states.shape[1]].unsqueeze(-1))
                j +=1
                episode_states = torch.cat([episode_states, state.detach().clone().unsqueeze(0)], dim=1)

                # Autoregressive action prediction
                sampled_action = action_sample(policy_model, episode_states, 1, temperature=1.0, sample=True, 
                        actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0),
                        rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1),
                        timesteps=t[:,:episode_states.shape[1]].unsqueeze(-1),
                        atgs=atgs, traj_len=traj_len)

                # ATG prompting only done
                if args.model_type != 'PromptDTATG':
                        atg = atgs[:,-1,:] + torch.ones_like(atg)*(args.perc/100)
                        atg[atg < 0] = 0
                        atgs = torch.cat([atgs, atg], dim=1)                        

        episode_states = episode_states.detach().cpu().squeeze(0).numpy().tolist()
        predicted_states.append(episode_states)

    return predicted_states


def main(args):

    # Load data
    test_path = os.path.join(args.datadir, 'test_Phys45.pickle')
    test_dataset = mimic_dataset.MimicTrajectoryDataset(test_path, CONTEXT_LENGTH, RTG_SCALE)    
    test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=True,
                                batch_size=1,
                                num_workers=2)

    # State predictor network architecture variables  
    sp_dir = os.path.join(args.log_dir, args.sp_dir)
    sp_net_dir = os.path.join(sp_dir, args.sp_dir+'_best_model.pt')
    sp_LAYERS = int(sp_dir.split('_')[2][6:])
    sp_EMB = int(sp_dir.split('_')[4][4:])    

    # Context length for different models
    sub_blocks = {'BC':CONTEXT_LENGTH*2, 'DT':CONTEXT_LENGTH*3, 'MeDT':CONTEXT_LENGTH*10, 
                  'PromptDT':CONTEXT_LENGTH*2+1, 'PromptDTATG':CONTEXT_LENGTH*2+8}

    # Policy network architecture variables
    pol_dir = os.path.join(args.log_dir, args.policy_dir)
    pol_net_dir = os.path.join(pol_dir, args.policy_dir+'_best_model.pt')
    pol_LAYERS = int(args.policy_dir.split('_')[2][6:])
    pol_EMB = int(args.policy_dir.split('_')[4][4:])

    # Load state predictor network
    block_size = CONTEXT_LENGTH*2
    mconf = GPTConfig(STATE_VOCAB_SIZE, block_size, 
            n_layer=sp_LAYERS, n_head=N_HEAD, n_embd=sp_EMB, model_type='state', 
            max_timestep=CONTEXT_LENGTH)
    sp_model = State_Predictor(mconf)
    sp_model = torch.nn.DataParallel(sp_model).to(device)
    sp_model.load_state_dict(torch.load(sp_net_dir, map_location=device))
    sp_model.eval()

    # Load policy network
    block_size = sub_blocks[args.model_type]
    mconf = GPTConfig(ACTION_VOCAB_SIZE, block_size, 
            n_layer=pol_LAYERS, n_head=N_HEAD, n_embd=pol_EMB, model_type=args.model_type, 
            max_timestep=CONTEXT_LENGTH)
    policy_model = MeDT(mconf)
    policy_model = torch.nn.DataParallel(policy_model).to(device)
    policy_model.load_state_dict(torch.load(pol_net_dir, map_location=device))
    policy_model.eval()

    # Evaluate policy model for patients in the test set for 10 steps
    states = evaluate_model(policy_model, sp_model, args.rtg, test_loader)

    # Calculate SAPS2 on estimated states
    scores = calculate_scores(states)

    # Compute mean and std of saps scores
    clc_mean = lambda scores, c: np.mean(np.array(scores)[:,:,c])
    calc_std = lambda scores, c: np.std(np.array(scores)[:,:,c]) / np.sqrt(np.size(np.array(scores)[:,:,c]))

    print('mean SAPS2 = {:.2f}+-{:.2f}'.format(clc_mean(scores, 0), calc_std(scores, 0)))
    print('Cardivascular: {:.2f}+-{:.2f}'.format(clc_mean(scores, 1), calc_std(scores, 1)))
    print('Respiratory: {:.2f}+-{:.2f}'.format(clc_mean(scores, 2), calc_std(scores, 2)))
    print('Neurological:{:.2f}+-{:.2f}'.format(clc_mean(scores, 3), calc_std(scores, 3)))
    print('Renal: {:.2f}+-{:.2f}'.format(clc_mean(scores, 4), calc_std(scores, 4)))
    print('Hepatic: {:.2f}+-{:.2f}'.format(clc_mean(scores, 5), calc_std(scores, 5)))
    print('Haematologic: {:.2f}+-{:.2f}'.format(clc_mean(scores, 6), calc_std(scores, 6)))
    print('Other: {:.2f}+-{:.2f}'.format(clc_mean(scores, 7), calc_std(scores, 7)))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default=".", type=str)
    parser.add_argument('--log_dir', default='./medDT/logs')
    parser.add_argument("--sp_dir", "-sp", 
                        default="SP_ID0_nLayer4_nHead8_nEmb64_LR0.0006_BS128_SEED0", help="")
    parser.add_argument("--policy_dir", "-p", 
                        default="MeDTv1_ID0_nLayer4_nHead8_nEmb64_LR0.0006_BS128_SEED0", help="") 
    parser.add_argument("--model_type", "-m", default="MeDT", type=str, help="Model type here")
    parser.add_argument("--rtg", default=1, type=int)
    parser.add_argument("--perc", default=0, type=float)
    args = parser.parse_args()
    
    main(args)