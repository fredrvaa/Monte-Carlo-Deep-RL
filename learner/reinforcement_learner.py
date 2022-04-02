from typing import Optional

import numpy as np

from environments.environment import Environment, Player
from learner.actor import Actor
from monte_carlo.monte_carlo import MonteCarloTree


class ReinforcementLearner:
    def __init__(self,
                 environment: Environment,
                 actor: Actor,
                 checkpoint_iter: Optional[int] = None
                 ):

        self.environment: Environment = environment
        self.actor: Actor = actor

        self.checkpoint_iter: Optional[int] = checkpoint_iter

    def fit(self, n_games: int = 100, n_search_games: int = 500, batch_size: int = 20, visualize: bool = False):
        replay_states = []
        replay_dists = []

        for n in range(n_games):
            if n % self.checkpoint_iter == 0:
                self.actor.checkpoint(f'{self.environment}_actor_{n}')

            print(f'----Game {n+1}----')

            state = self.environment.initialize(np.random.choice(list(Player)))
            mct = MonteCarloTree(self.environment, self.actor.get_lite_model(), M=n_search_games)
            print(f'Building replay buffer...', end='')
            final, winning_player = False, None
            while not final:
                print('.', end='')
                dist = mct.simulation(state)

                replay_states.append(state)
                replay_dists.append(dist)

                action = int(np.argmax(dist))
                final, winning_player, state = self.environment.step(state, action, visualize=visualize)

            print(f'\nFitting model...')

            fit_idxs = np.random.choice(len(replay_states), size=batch_size)
            self.actor.model.fit(
                x=np.array(replay_states)[fit_idxs],
                y=np.array(replay_dists)[fit_idxs],
            )
            show = np.random.choice(fit_idxs)
            print(replay_states[show])
            print(replay_dists[show])

        self.actor.checkpoint(f'{self.environment}_actor_{n_games}')
