"""
This is a utility script used to run a Tournament of Progressive Policies from a config file.

Usage: python run_topp.py -f <path/to/models/folder> -c <path/to/topp/config>
"""

import os
import argparse

import matplotlib.pyplot as plt

from config.parser import ToppParser
from environments.environment import Environment
from environments.hex import Hex
from topp.topp import Topp
from topp.visualize import plot_win_ratios

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, help='path/to/models/folder', default='models')
parser.add_argument('-c', '--config', type=str, help='path/to/topp/config')
args = parser.parse_args()

# Get model paths sorted by games played
model_paths = [f'{args.folder}/{model_name}' for model_name in os.listdir(args.folder) if model_name.endswith('.h5')]
model_paths.sort(key=lambda p: int(p.split('.')[0].split('_')[-1]))
model_names = [p.split('.')[0].split('/')[-1] for p in model_paths]
print('Models: ', model_names)

# Parse config
configparser = ToppParser(args.config)
environment: Environment = configparser.get_environment()

# Run tournament
topp = Topp(environment, model_paths)
round_robin_kwargs = configparser.get_round_robin_kwargs()
wins = topp.round_robin(**round_robin_kwargs)

# Plot win ratios per model
win_ratios = wins / round_robin_kwargs['n_games']
print('Win ratios: ', win_ratios.sum(axis=1) / (len(model_names) - 1))
title = 'Probabilistic TOPP' if round_robin_kwargs['probabilistic'] else 'Deterministic TOPP'
plot_win_ratios(model_names=model_names, win_ratios=win_ratios, title=title)

if round_robin_kwargs['visualize'] and isinstance(environment, Hex):
    plt.show()