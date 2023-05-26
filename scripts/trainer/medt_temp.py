"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import wandb	
from torch.utils.tensorboard import SummaryWriter
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from collections import deque
import random
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:


    def __init__(self, model, train_dataset, eval_dataset, test_dataset, exp_name, config):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.exp_name = exp_name

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def run_epoch(self, split, best_val_loss, epoch_num=0):

        is_train = split == 'train'
        self.model.train(is_train)

        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=self.args.batchsize,
                            num_workers=self.args.num_workers)

        eval_loader = DataLoader(self.eval_dataset, shuffle=True, pin_memory=True,
                            batch_size=self.args.batchsize,
                            num_workers=self.args.num_workers)

        losses = []
        accuracies = []
        val_losses = []
        val_accuracies = []

        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)	
        eval_pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
        

        correct = 0
        total_samples = 0

        for it, (x, y, r, t, saps, divSaps) in pbar:
            
            self.model.train()
            # states, actions, rtgs, timesteps
            # place data on the correct device
            x = x.to(self.device)
            y = y.unsqueeze(-1).to(self.device)
            r = r.unsqueeze(-1).to(self.device)
            t = t.unsqueeze(-1).to(self.device)
            saps = saps.unsqueeze(-1).to(self.device)
            divSaps = divSaps.to(self.device)

            
            # forward the model
            with torch.set_grad_enabled(is_train):
                # logits, loss = model(x, y, r)
                logits, loss, _ = self.model(x, y, y, r, t, saps, divSaps)  # states, actions, targets, rtgs, timesteps, saps
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                
                probs = torch.softmax(logits,dim=-1)
                out = probs.argmax(dim = -1)

                y = y.squeeze(-1)

                correct += (out == y).sum().float().detach().cpu()
                total_samples += float(out.shape[0]*out.shape[1])

            if is_train:

                # backprop and update the parameters
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                losses.append(loss.detach().cpu().item())

                # decay the learning rate based on our progress
                if self.args.lr_decay:
                    self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < self.config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - self.config.warmup_tokens) / float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.config.learning_rate

            mean_train_accuracy =  correct/total_samples
            self.model.eval()

            correct = 0
            total_samples = 0

            for it, (x, y, r, t, saps, divSaps) in eval_pbar:
                
                # states, actions, rtgs, timesteps
                # place data on the correct device
                x = x.to(self.device)
                y = y.unsqueeze(-1).to(self.device)
                r = r.unsqueeze(-1).to(self.device)
                t = t.unsqueeze(-1).to(self.device)
                saps = saps.unsqueeze(-1).to(self.device)
                divSaps = divSaps.to(self.device)
          
                val_logits, val_loss, _ = self.model(x, y, y, r, t, saps, divSaps)  #states, actions, targets, rtgs, timesteps
                val_loss = val_loss.mean() # collapse all losses if they are scattered on multiple gpus
                val_losses.append(val_loss.detach().cpu().item())
              
                probs = torch.softmax(val_logits,dim=-1)
                out = probs.argmax(dim = -1)

                y = y.squeeze(-1)
                correct += (out == y).sum().float().detach().cpu()
                total_samples += float(out.shape[0]*out.shape[1])
                

            mean_val_accuracy = correct/total_samples

            mean_train_loss = sum(losses)/len(losses)
            mean_val_loss = sum(val_losses)/len(val_losses)

            self.df.loc[epoch_num] = [epoch_num, mean_train_loss, mean_val_loss, mean_train_accuracy, mean_val_accuracy]

            self.e_losses.append(mean_train_loss)
            self.e_accuracies.append(mean_train_accuracy)

            self.e_val_losses.append(mean_val_loss)
            self.e_val_accuracies.append(mean_val_accuracy)            

            if mean_val_loss < self.best_val_loss:
                self.best_val_loss = mean_val_loss
                torch.save(self.model.state_dict(), self.save_best_model_path) 
                print('best model saved')

            if epoch_num % 10 == 0:
                ep_model_path =  os.path.join(self.log_dir, '{}_Epoch{}_model.pt'.format(self.exp_name, epoch_num))
                torch.save(self.model.state_dict(), ep_model_path)

            # report progress
            pbar.set_description(f"epoch {epoch_num+1} iter {it}: train loss {mean_train_loss:.5f}, val loss {mean_val_loss:.5f}.. lr {lr:e}")
            print(f"epoch {epoch_num+1} iter {it}: train loss {mean_train_loss:.5f}, val loss {mean_val_loss:.5f}, train accuracy {mean_train_accuracy:.5f}, val accuracy {mean_val_accuracy:.5f}. lr {lr:e}")


    def train(self, args):
        
        self.args = args	
        	
        os.environ["WANDB_API_KEY"] = "7a9cbed74d12db3de9cef466bb7b7cf08bdf1ea4"	
        os.environ["WANDB_MODE"] = "offline"	
        wandb_config = {	
            "machine": "Compute Canada",	
            "model": "MeDT",	
            "learning_rate": args.learningrate,	
            "batch_size": args.batchsize,	
        }	
        wandb.init(project='MeDTv1')
        
        lst = ['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
 
        self.df = pd.DataFrame(columns = lst)
        
        self.log_dir = os.path.join(args.logdir, self.exp_name)
        
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        self.save_best_model_path =  os.path.join(self.log_dir, '{}_best_model.pt'.format(self.exp_name))
        self.loss_plot_path = os.path.join(self.log_dir, '{}_loss_plot.png'.format(self.exp_name))
        self.accuracy_plot_path = os.path.join(self.log_dir, '{}_accuracy_plot.png'.format(self.exp_name))
        
        model, config = self.model, self.config
        wandb.watch(self.model, log_freq=1)
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer = raw_model.configure_optimizers(config)
                        
        best_return = -float('inf')
        wandb.watch(model, log_freq=1)
        self.tokens = 0 # counter used for learning rate decay

        self.e_losses = []
        self.e_accuracies = []
        self.e_val_losses = []
        self.e_val_accuracies = []        
        
        self.best_val_loss = float('inf')
        for epoch in range(args.epochs):
            self.run_epoch('train', self.best_val_loss,epoch_num=epoch)


        csv_saveloc = os.path.join(log_dir, 'out.csv')

        self.df.to_csv('{}.csv'.format(csv_saveloc))