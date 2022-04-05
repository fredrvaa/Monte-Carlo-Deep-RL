"""
Contains the core reinforcement learning algorithm.
"""

import time
import datetime
import numpy as np
from typing import Optional

from environments.environment import Environment, Player
from learner.actor import Actor
from learner.replay_buffer import ReplayBuffer
from monte_carlo.monte_carlo_tree import MonteCarloTree

import tensorflow.keras.backend as K


class ReinforcementLearner:
    """
    Class used to train an actor using data gathered from monte carlo simulations.
    """

    def __init__(self,
                 environment: Environment,
                 actor: Actor,
                 ):
        """
        :param environment: Environment the actor should be fitted to
        :param actor: The actor to fit
        """

        self.environment: Environment = environment
        self.actor: Actor = actor

    def fit(self,
            n_games: int = 100,
            n_search_games: int = 500,
            batch_size: int = 20,
            buffer_size: Optional[int] = None,
            epochs: int = 1,
            visualize: bool = False,
            vis_delay: float = 0.1,
            checkpoint_iter: Optional[int] = None) -> None:
        """
        Fits the actor to the environment by supervised learning.

        Data is incrementally gathered by performing monte carlo simulations and adding (state, action_distribution)
        pairs to a replay buffer. Random actions are taken after each monte carlo simulation in order to
        maintain diversity in the replay buffer.

        Mini batches from the replay buffer is then used to fit the actor.

        :param n_games: Number of real games to perform RL for
        :param n_search_games: Number of search games in each monte carlo simulation
        :param batch_size: Batch size retrieved from the replay buffer to be used to train the actor
        :param buffer_size: Max buffer size. If None, the buffer size is unlimited
        :param epochs: Number of epochs to train the actor for each minibatch from the replay buffer
        :param visualize: Whether to visualize each game
        :param vis_delay: Delay between each visualization
        :param checkpoint_iter: Number of iterations between checkpointing actor model to file
        """

        rbuf = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size)

        for n in range(n_games):
            start_time = time.time()
            if checkpoint_iter is not None and n % checkpoint_iter == 0:
                self.actor.checkpoint(f'{self.environment}_actor_{n}')

            print(f'----Game {n+1}----')

            state = self.environment.initialize(np.random.choice(list(Player)))
            if visualize:
                self.environment.visualize(state, vis_delay, n)
            mct = MonteCarloTree(self.environment, self.actor.get_lite_model(), root_state=state, M=n_search_games)
            print('Building replay buffer...')
            final, winning_player = False, None

            while not final:
                dist = mct.simulation()

                rbuf.store_replay(state, dist)

                action = self.environment.get_random_action(state)
                final, winning_player, state = self.environment.step(state, action)
                if visualize:
                    self.environment.visualize(state, vis_delay, n)
                mct.set_new_root(action)

            print(f'Buffer size: {rbuf.n}')

            mct_time = time.time()
            print('Fitting model...')

            if rbuf.is_ready:
                x, y = rbuf.get_batch()
                self.actor.model.fit(x, y, epochs=epochs)
                print('Learning rate: ', K.eval(self.actor.model.optimizer._decayed_lr(float)))

            end_time = time.time()
            print('MCTS time: ', datetime.timedelta(seconds=mct_time - start_time))
            print('NN time: ', datetime.timedelta(seconds=end_time - mct_time))
            print('Total time: ', datetime.timedelta(seconds=end_time - start_time))

        self.actor.checkpoint(f'{self.environment}_actor_{n_games}')
