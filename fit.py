import os
import argparse

from config.parser import FitParser
from environments.environment import Environment
from learner.actor import Actor

from learner.reinforcement_learner import ReinforcementLearner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path/to/fit/config', type=str)
args = parser.parse_args()

# Parse config
configparser = FitParser(args.config)

environment: Environment = configparser.get_environment()
actor_kwargs = configparser.get_actor_kwargs()
actor: Actor = Actor(input_size=environment.state_size, output_size=environment.n_actions, **actor_kwargs)

# Run reinforcement learning
rl = ReinforcementLearner(environment, actor)
fit_kwargs = configparser.get_fit_kwargs()
rl.fit(**fit_kwargs)
