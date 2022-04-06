"""
This is a utility script used to train a actor NN with RL and MCTS from a config file.

Usage: python fit.py -c <path/to/fit/config>
"""

import os
import argparse
from typing import Optional

from config.parser import FitParser
from environments.environment import Environment

from learner.network import Network
from learner.reinforcement_learner import ReinforcementLearner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path/to/fit/config', type=str)
args = parser.parse_args()

# Parse config and load environment, actor, and critic
configparser = FitParser(args.config)

environment: Environment = configparser.get_environment()

actor_kwargs = configparser.get_network_kwargs('actor')
actor: Network = Network(name='Actor',
                         input_size=environment.state_size,
                         output_size=environment.n_actions,
                         **actor_kwargs)

try:
    critic_kwargs = configparser.get_network_kwargs('critic')
    critic: Optional[Network] = Network(name='Critic',
                              input_size=environment.state_size,
                              output_size=1,
                              **critic_kwargs)
except KeyError:
    critic: Optional[Network] = None


# Run reinforcement learning
rl = ReinforcementLearner(environment, actor, critic)
fit_kwargs = configparser.get_fit_kwargs()
rl.fit(**fit_kwargs)
