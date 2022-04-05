import os
import argparse

from config.parser import ToppParser
from environments.environment import Environment
from topp.topp import Topp
from topp.visualize import plot_win_ratios

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, help='path/to/models/folder', default='models')
parser.add_argument('-c', '--config', type=str, help='path/to/topp/config')
args = parser.parse_args()

# Get model paths sorted by games played
model_paths = [f'{args.folder}/{model_name}' for model_name in os.listdir(args.folder)]
model_paths.sort(key=lambda p: int(p.split('.')[0].split('_')[-1]))
model_names = [p.split('.')[0].split('/')[-1] for p in model_paths]

# Parse config
configparser = ToppParser(args.config)
environment: Environment = configparser.get_environment()

# Run tournament
topp = Topp(environment, model_paths)
round_robing_kwargs = configparser.get_round_robin_kwargs()
wins = topp.round_robin(**round_robing_kwargs)

# Plot win ratios per model
win_ratios = wins / round_robing_kwargs['n_games']
print('Total wins: ', wins.sum(axis=1))
print('Win ratios: ', win_ratios.mean(axis=1))
plot_win_ratios(model_names, win_ratios)