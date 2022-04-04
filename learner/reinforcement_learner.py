import time
import datetime
import numpy as np
from typing import Optional

from environments.environment import Environment, Player
from learner.actor import Actor
from learner.replay_buffer import ReplayBuffer
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
        rbuf = ReplayBuffer(batch_size=batch_size)

        for n in range(n_games):
            start_time = time.time()
            if n % self.checkpoint_iter == 0:
                self.actor.checkpoint(f'{self.environment}_actor_{n}')

            print(f'----Game {n+1}----')

            state = self.environment.initialize(np.random.choice(list(Player)))
            mct = MonteCarloTree(self.environment, self.actor.get_lite_model(), root_state=state, M=n_search_games)
            print('Building replay buffer...')
            final, winning_player = False, None
            while not final:
                dist = mct.simulation()

                rbuf.store_replay(state, dist)

                action = int(np.argmax(dist))
                final, winning_player, state = self.environment.step(state, action, visualize=visualize)
                mct.set_new_root(action)

            ## NIM DEBUG START
            x, y = rbuf.get_sample()
            print(x, self.environment._to_value(x))
            print(y, np.argmax(y) + 1)
            ## NIM DEBUG END

            mct_time = time.time()
            print('Fitting model...')

            if rbuf.is_ready:
                x, y = rbuf.get_batch()
                self.actor.model.fit(x, y)

            end_time = time.time()
            print('MCTS time: ', datetime.timedelta(seconds=mct_time - start_time))
            print('NN time: ', datetime.timedelta(seconds=end_time - mct_time))
            print('Total time: ', datetime.timedelta(seconds=end_time - mct_time))

        self.actor.checkpoint(f'{self.environment}_actor_{n_games}')
