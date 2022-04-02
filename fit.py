from environments.hex import Hex
from environments.nim import Nim
from learner.actor import Actor
from learner.reinforcement_learner import ReinforcementLearner

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#environment = Hex(k=7)
environment = Nim(starting_stones=15, max_take=5)

actor = Actor(input_size=environment.state_size,
              output_size=environment.n_actions,
              hidden_sizes=[64, 128, 64],
              learning_rate=1e-4,
              checkpoint_folder='models'
              )

rl = ReinforcementLearner(environment, actor, checkpoint_iter=25)
rl.fit(200, 2000, 200)
