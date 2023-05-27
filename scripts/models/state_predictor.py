import torch
import torch.nn as nn
from torch.nn import functional as F

from models.GPT import GPT


################################################################################################################
# class State_Predictor
#
# We make embeddings of each input in the sequence. We then add position embeddings and feed embeddings
# to GPT transformer to predict states
#       
#
################################################################################################################
class State_Predictor(GPT):
   
    # state, action
    def forward(self, states, actions, targets=None, timesteps=None):
        # states: (batch, context_len, 45)
        # actions: (batch, block_size, 1)
        # targets: (batch, context_len-1, 45)
        # timesteps: (batch, 1, 1)

        state_embeddings = self.state_emb(states.type(torch.float32))
        action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

        token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
        token_embeddings[:,::2,:] = state_embeddings
        token_embeddings[:,1::2,:] = action_embeddings#[:,-states.shape[1]:,:] #+ int(targets is None):,:]   
       
        my_pos_emb = torch.zeros(timesteps.shape[0] ,timesteps.shape[1]*2, self.config.n_embd).to(self.device)
        my_pos_emb[:,0::2,:] = timesteps
        my_pos_emb[:,1::2,:] = timesteps
       
        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]  + my_pos_emb
        x = self.drop(token_embeddings) + position_embeddings

        for idx, block in enumerate(self.blocks):
            x, attn_score = block(x)


        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        logits = logits[:, 1::2, :]
        loss = None


        if targets is not None:
            loss = F.mse_loss(logits[:,:-1,:], targets[:,1:,:])

        return logits, loss