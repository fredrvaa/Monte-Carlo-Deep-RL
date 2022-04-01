import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from environments.environment import Environment
from environments.hex import Hex

from learner.reinforcement_learner import ReinforcementLearner

environment = Hex(k=7)
rl = ReinforcementLearner(environment, [256, 128, 64], checkpoint_iter=5)
rl.fit(100, 1000, 50)