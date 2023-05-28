import os
import argparse

import torch

from datasets import mimic_dataset
from models.GPT import GPTConfig, GPT
from models.state_predictor import State_Predictor
from trainer.trainer_SP import TrainerConfig, Trainer


################################################################################################################
# Constants
#
################################################################################################################
CONTEXT_LENGTH = 20
RTG_SCALE = 1
VOCAB_SIZE = 45
WARMPUP_TOKENS = 512*20
BLOCK_SIZE = CONTEXT_LENGTH*2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################################################
# train
#
# This is where learning happens. More specifically, the state predictor network is trained to estimate states  
# with teacher forcing using MSE loss.
#
#
################################################################################################################
def train(args):

    # Experiment name
    exp_name = f'SP_ID{args.id}_nLayer{args.n_layer}_nHead{args.n_head}_nEmb{args.n_embd}_LR{args.learningrate}_BS{args.batchsize}_SEED{args.seed}'

    # Load dataset
    train_dataset = mimic_dataset.MimicTrajectoryDataset(os.path.join(args.datadir, 'train_Phys45.pickle'), CONTEXT_LENGTH, RTG_SCALE)
    eval_dataset = mimic_dataset.MimicTrajectoryDataset(os.path.join(args.datadir, 'val_Phys45.pickle'), CONTEXT_LENGTH, RTG_SCALE)
    
    # Build model
    mconf = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, 
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, model_type='state', max_timestep=CONTEXT_LENGTH)
    model = State_Predictor(mconf)

    # Train model
    tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batchsize, learning_rate=args.learningrate,
                        lr_decay=args.lr_decay, warmup_tokens=WARMPUP_TOKENS, final_tokens=2*len(train_dataset)*CONTEXT_LENGTH*2,
                        num_workers=args.num_workers, seed=args.seed, model_type='state', max_timestep=CONTEXT_LENGTH)
    trainer = Trainer(model, train_dataset, eval_dataset, test_dataset, exp_name, tconf)
    trainer.train(args=args)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./", type=str)
    parser.add_argument("--epochs", "-e", type=int, default=2000)
    parser.add_argument("--batchsize", "-b", type=int, default=128)
    parser.add_argument("--learningrate", "-lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="./medDT/logs")
    parser.add_argument("--id", type=str, default="0")
    parser.add_argument("--loadfile", "-l", action="store_true")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--num_workers", "-nw", type=int, default=4)
    parser.add_argument("--lr_decay", "-ld", action="store_true")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=128)
    args = parser.parse_args()

    train(args)