import wandb
import sys
import os
import argparse
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train

if False:
    sweep_id = "o83xj677"
    wandb.agent(sweep_id, function=train, project="hypertune", count=1)
else:

    parser = argparse.ArgumentParser(description='Start a wandb agent with a sweep ID')
    parser.add_argument('--sweep_id', type=str, help='Sweep ID from wandb')
    parser.add_argument('--missions', type=int, help='Number of missions to run')
    args = parser.parse_args()

    sweep_id = args.sweep_id
    missions = args.missions
    wandb.agent(sweep_id, function=train, count=missions, project="hypertune")