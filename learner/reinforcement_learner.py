from typing import Optional

import numpy as np

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from environments.environment import Environment, Player
from environments.hex import Hex
from learner.lite_model import LiteModel
from monte_carlo.monte_carlo import MonteCarloTree


class ReinforcementLearner:
    def __init__(self,
                 environment: Environment,
                 hidden_sizes: list[int],
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 checkpoint_iter: Optional[int] = 5,
                 checkpoint_folder: Optional[str] = 'models'
                 ):

        self.environment: Environment = environment
        state = self.environment.initialize()
        self.actor = tfk.Sequential()
        self.actor.add(tfkl.InputLayer(input_shape=state.shape))
        for size in hidden_sizes:
            self.actor.add(tfkl.Dense(units=size, activation=activation))
        self.actor.add(tfkl.Dense(units=self.environment.n_actions))
        self.actor.add(tfkl.Softmax())

        self.actor.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=optimizer)
        self.actor.summary()

        self.checkpoint_iter: Optional[int] = checkpoint_iter
        self.checkpoint_folder: Optional[str] = checkpoint_folder

    def _checkpoint_model(self, n):
        if self.checkpoint_folder is not None and self.checkpoint_iter is not None:
            tfk.models.save_model(self.actor, f'{self.checkpoint_folder}/{self.environment.__name__}actor_{n}.h5')

    def fit(self, n_games: int = 100, n_search_games: int = 500, batch_size: int = 20, visualize: bool = False):
        replay_states = []
        replay_dists = []

        for n in range(n_games):
            self._checkpoint_model(n)
            print(f'----Game {n+1}----')

            state = self.environment.initialize(np.random.choice(list(Player)))
            mct = MonteCarloTree(self.environment, LiteModel.from_keras_model(self.actor), M=n_search_games)
            print(f'Building replay buffer...')
            final, winning_player = False, None
            while not final:
                dist = mct.simulation(state)
                replay_states.append(state)
                replay_dists.append(dist)

                action = int(np.argmax(dist))
                final, winning_player, state = self.environment.step(state, action, visualize=visualize)
            print(replay_dists[-1])
            print(f'Fitting model...')
            print(len(replay_states))

            self.actor.fit(
                x=np.array(replay_states),
                y=np.array(replay_dists),
                batch_size=batch_size,
                epochs=50)

        self._checkpoint_model(n)


if __name__ == '__main__':
    environment = Hex(k=7)
    rl = ReinforcementLearner(environment, [256, 128, 64], checkpoint_iter=5)
    rl.fit(100, 1000, 50)
