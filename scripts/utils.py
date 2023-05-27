import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


################################################################################################################
# Autoregressive sampling of states
#
# Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
# the sequence, feeding the predictions back into the model each time.
#
# Inputs:
#   x: sequence of states
#   actions: sequence of actions
#   model: state predictor
#   
################################################################################################################
@torch.no_grad()
def state_sample(model, x, steps, actions=None, timesteps=None):
    block_size = 50
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size//2 else x[:, -block_size//2:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//2 else actions[:, -block_size//2:] # crop context if needed
        out, _ = model(x_cond, actions=actions, targets=None, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        out = out[:, -1, :]
    return out



################################################################################################################
# Autoregressive sampling of actions
#
# Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
# the sequence, feeding the predictions back into the model each time.
#
# Inputs:
#   x: sequence of states
#   actions: sequence of actions
#   model: policy model
#   rtg: returns-to-go conditioning
#   atg: acuity-to-go conditioning
#   
################################################################################################################
@torch.no_grad()
def action_sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None, atgs=None, traj_len=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. 
    """
    block_size = 200
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size//10 else x[:, -block_size//10:] # crop context if needed
        
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//10 else actions[:, -block_size//10:] # crop context if needed

        atgs = atgs if atgs.size(1) <= block_size//10 else atgs[:, -block_size//10:] # crop context if needed            
        rtgs = rtgs if rtgs.size(1) <= block_size//10 else rtgs[:, -block_size//10:] # crop context if needed

        logits, _, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps, divSaps=atgs, traj_len=traj_len)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = ix
    return x


################################################################################################################
# SAPS2 calculator function
#
# Take a state of dim 45 and calculate the SAPS2 acuity score
#
# Inputs:
#   x: sequence of states
#   actions: sequence of actions
#   model: policy model
#   rtg: returns-to-go conditioning
#   atg: acuity-to-go conditioning
#   
################################################################################################################
def calculate_saps2(state):   
    
    # Map names to patient state
    col_names = ['o:GCS', 'o:HR', 'o:SysBP',
          'o:MeanBP', 'o:DiaBP', 'o:RR', 'o:Temp_C', 'o:FiO2_1', 'o:Potassium',
          'o:Sodium', 'o:Chloride', 'o:Glucose', 'o:Magnesium', 'o:Calcium',
          'o:Hb', 'o:WBC_count', 'o:Platelets_count', 'o:PTT', 'o:PT',
          'o:Arterial_pH', 'o:paO2', 'o:paCO2', 'o:Arterial_BE', 'o:HCO3',
          'o:Arterial_lactate', 'o:SIRS', 'o:Shock_Index',
          'o:PaO2_FiO2', 'o:cumulated_balance', 'o:SpO2', 'o:BUN', 'o:Creatinine',
          'o:SGOT', 'o:SGPT', 'o:Total_bili', 'o:INR', 'o:input_total',
          'o:input_4hourly', 'o:output_total', 'o:output_4hourly','o:gender', 
          'o:mechvent', 'o:re_admission', 'o:age', 'o:Weight_kg']
    state = dict(zip(col_names, state))
    
    # Covariate intervals for score calculation
    age_values = np.array([0, 7, 12, 15, 16, 18])
    hr_values = np.array([11, 2, 0, 4, 7])
    bp_values = np.array([13, 5, 0, 2])
    temp_values = np.array([0, 3])
    o2_values = np.array([11, 9, 6])
    output_values = np.array([11, 4, 0])
    bun_values = np.array([0, 6, 10])
    wbc_values = np.array([12, 0, 3])
    k_values = np.array([3, 0, 3])
    na_values = np.array([5, 0, 1])
    hco3_values = np.array([5, 3, 0])
    bili_values = np.array([0, 4, 9])
    gcs_values = np.array([26, 13, 7, 5, 0])

    # Calculate score for each covariate
    age = np.array([ state['o:age']<40, (state['o:age']>=40)&(state['o:age']<60), (state['o:age']>=60)&(state['o:age']<70), (state['o:age']>=70)&(state['o:age']<75), (state['o:age']>=75)&(state['o:age']<80), state['o:age']>=80 ])
    hr = np.array([ state['o:HR']<40, (state['o:HR']>=40)&(state['o:HR']<70), (state['o:HR']>=70)&(state['o:HR']<120), (state['o:HR']>=120)&(state['o:HR']<160), state['o:HR']>=160 ])
    bp = np.array([ state['o:SysBP']<70, (state['o:SysBP']>=70)&(state['o:SysBP']<100), (state['o:SysBP']>=100)&(state['o:SysBP']<200), state['o:SysBP']>=200 ])
    temp = np.array([ state['o:Temp_C']<39, state['o:Temp_C']>=39 ])
    o2 = np.array([ state['o:PaO2_FiO2']<100, (state['o:PaO2_FiO2']>=100)&(state['o:PaO2_FiO2']<200), state['o:PaO2_FiO2']>=200 ])
    out = np.array([ state['o:output_4hourly']<500, (state['o:output_4hourly']>=500)&(state['o:output_4hourly']<1000), state['o:output_4hourly']>=1000 ])
    bun = np.array([ state['o:BUN']<28, (state['o:BUN']>=28)&(state['o:BUN']<84), state['o:BUN']>=84 ])
    wbc = np.array([ state['o:WBC_count']<1, (state['o:WBC_count']>=1)&(state['o:WBC_count']<20), state['o:WBC_count']>=20 ])
    k = np.array([ state['o:Potassium']<3, (state['o:Potassium']>=3)&(state['o:Potassium']<5), state['o:Potassium']>=5 ])
    na = np.array([ state['o:Sodium']<125, (state['o:Sodium']>=125)&(state['o:Sodium']<145), state['o:Sodium']>=145 ])
    hco3 = np.array([ state['o:HCO3']<15, (state['o:HCO3']>=15)&(state['o:HCO3']<20), state['o:HCO3']>=20 ])
    bili = np.array([ state['o:Total_bili']<4, (state['o:Total_bili']>=4)&(state['o:Total_bili']<6), state['o:Total_bili']>=6 ])
    gcs = np.array([ state['o:GCS']<6, (state['o:GCS']>=6)&(state['o:GCS']<9), (state['o:GCS']>=9)&(state['o:GCS']<11), (state['o:GCS']>=11)&(state['o:GCS']<14), state['o:GCS']>=14 ])
    
    # Calculate SAPS score
    sapsii = max(age_values[age], default=0) + max(hr_values[hr], default=0) + max(bp_values[bp], default=0) + max(temp_values[temp], default=0) + max(o2_values[o2]*state['o:mechvent'], default=0) + max(output_values[out], default=0) + max(bun_values[bun], default=0) + max(wbc_values[wbc], default=0) + max(k_values[k], default=0) + max(na_values[na], default=0) + max(hco3_values[hco3], default=0) + max(bili_values[bili], default=0) + max(gcs_values[gcs], default=0)

    # Calculate SAPS constituents
    Cardivascular = max(hr_values[hr], default=0) + max(bp_values[bp], default=0) + max(output_values[out], default=0)
    Respiratory =  max(o2_values[o2] * state['o:mechvent'], default=0)
    Neurological = max(gcs_values[gcs], default=0)
    Renal =  max(bun_values[bun], default=0)
    Hepatic = max(bili_values[bili], default=0)
    Haematologic = max(wbc_values[wbc], default=0)
    Other = max(temp_values[temp], default=0) + max(k_values[k], default=0) + max(na_values[na], default=0)

    return sapsii, Cardivascular, Respiratory, Neurological, Renal, Hepatic, Haematologic, Other


################################################################################################################
# Calculates SAPS scores over patient state trajectories 
#
# Take a trajectories of dim (n,t,45)
#
# where:
#   n: number of trajectories
#   t: length of each trajectory
#   
################################################################################################################
def calculate_scores(states):
    saps2_scores = []
    
    # Iterate over number of trajectories
    for i in range(len(states)):
        saps2_traj = []
        
        # Iterate over each state in trajectory
        for j in range(len(states[i])):
            saps2_traj.append(calculate_saps2(states[i][j]))
        saps2_scores.append(saps2_traj)     
    
    return saps2_scores


# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)