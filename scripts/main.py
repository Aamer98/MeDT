import os
import argparse
import wandb
import torch

from datasets import mimic_dataset
from models.model import GPTConfig, GPT
from train_model import TrainerConfig, Trainer

################################################################################################################
# Constants
#
################################################################################################################
CONTEXT_LENGTH = 20
RTG_SCALE = 1
VOCAB_SIZE = 25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    # Load dataset
    train_path = os.path.join(args.datadir, 'train_Phys45.pickle')
    eval_path = os.path.join(args.datadir, 'val_Phys45.pickle')
    test_path = os.path.join(args.datadir, 'test_Phys45.pickle')

    train_dataset = mimic_dataset.MimicTrajectoryDataset(train_path, CONTEXT_LENGTH, RTG_SCALE)
    eval_dataset = mimic_dataset.MimicTrajectoryDataset(eval_path, CONTEXT_LENGTH, RTG_SCALE)
    test_dataset = mimic_dataset.MimicTrajectoryDataset(test_path, CONTEXT_LENGTH, RTG_SCALE)

    # Build model
    block_size = CONTEXT_LENGTH*10
    mconf = GPTConfig(VOCAB_SIZE, block_size,
                    n_layer=6, n_head=8, n_embd=128, model_type='reward_Physaps_conditioned', max_timestep=20)
    model = GPT(mconf)

    # Train model
    tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batchsize, learning_rate=args.learningrate,
                        lr_decay=True, warmup_tokens=args.warmup_tokens, final_tokens=2*len(train_dataset)*CONTEXT_LENGTH*10,
                        num_workers=4, seed=args.seed, model_type='reward_Physaps_conditioned', max_timestep=20)
    trainer = Trainer(model, train_dataset, eval_dataset, test_dataset, 'Div45A_medDT_rtg_Physaps_V1', tconf)
    trainer.train(logs_dir=args.logdir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/home/aamer98/scratch/Datasets/mimic_sepsis", type=str)
    parser.add_argument("--epochs", "-e", type=int, default=250)
    parser.add_argument("-batchsize", "-b", type=int, default=128)
    parser.add_argument("--learningrate", "-lr", type=float, default=6e-4)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--warmup_tokens", "-wt", type=float, default=512*20)
    parser.add_argument("--logdir", default="logs", type=str)
    parser.add_argument("--id", default="baseline", type=str)
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--save", "-s", action="store_true")
    args = parser.parse_args()
    
    main(args)