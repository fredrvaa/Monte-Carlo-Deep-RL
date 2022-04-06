"""
Contains the core reinforcement learning algorithm.
"""

import time
import datetime
import numpy as np
from typing import Optional

from environments.environment import Environment, Player

from learner.network import Network
from learner.replay_buffer import ReplayBuffer
from monte_carlo.monte_carlo_tree import MonteCarloTree

np.set_printoptions(precision=2, suppress=True)


class ReinforcementLearner:
    """
    Class used to train an actor using data gathered from monte carlo simulations.
    """

    def __init__(self,
                 environment: Environment,
                 actor: Network,
                 critic: Network,
                 start_epsilon: float = 1.0,
                 start_sigma: float = 1.0
                 ):
        """
        :param environment: Environment the actor should be fitted to
        :param actor: Actor to fit
        :param critic: Critic used to evaluate states during MCTS
        """

        self.environment: Environment = environment
        self.actor: Network = actor
        self.critic: Network = critic

        self.start_epsilon: float = start_epsilon
        self.start_sigma: float = start_sigma

    def fit(self,
            n_games: int = 100,
            n_search_games: int = 500,
            batch_size: int = 20,
            buffer_size: Optional[int] = None,
            epochs: int = 1,
            visualize: bool = False,
            vis_delay: float = 0.1,
            checkpoint_iter: Optional[int] = None,
            verbose: int = 2) -> None:
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
        :param verbose: Decides verbosity during fit: 0=minimal, 1=essential, 2=extra
        """

        rbuf = ReplayBuffer(targets=['actor', 'critic'], batch_size=batch_size, buffer_size=buffer_size)

        for n in range(n_games):
            epsilon = self.start_epsilon * (1 - (n/n_games))
            sigma = self.start_sigma * (1 - (n / n_games))
            start_time = time.time()
            if checkpoint_iter is not None and n % checkpoint_iter == 0:
                self.actor.checkpoint(f'{self.environment}_actor_{n}')

            lite_actor_model = self.actor.get_lite_model()
            lite_critic_model = self.critic.get_lite_model()

            print(f'----Game {n+1}----')
            if verbose > 1:
                print(f'Epsilon: {epsilon}, Sigma: {sigma}')
            state = self.environment.initialize(np.random.choice(list(Player)))
            visited_states = []
            actor_targets = []

            if visualize:
                self.environment.visualize(state, vis_delay, n)

            mct = MonteCarloTree(environment=self.environment,
                                 actor=lite_actor_model,
                                 critic=lite_critic_model,
                                 root_state=state,
                                 M=n_search_games)

            if verbose > 0:
                print('Building replay buffer...')
            final, winning_player = False, None

            while not final:
                dist = mct.simulation(epsilon=epsilon, sigma=sigma)

                visited_states.append(state)
                actor_targets.append(dist)

                if np.random.random() > epsilon:
                    dist = lite_actor_model.predict_single(state)
                    action = self.environment.get_action_from_distribution(state, dist)
                    #action = self.environment.get_action_from_distribution(state, dist)
                else:
                    action = self.environment.get_random_action(state)

                final, winning_player, state = self.environment.step(state, action)
                if visualize:
                    self.environment.visualize(state, vis_delay, n)
                mct.set_new_root(action)
            critic_targets = [1]*len(visited_states) if winning_player == Player.one else [-1]*len(visited_states)

            rbuf.store_replays(states=visited_states, actor=actor_targets, critic=critic_targets)

            if verbose > 1:
                print(f'Buffer size: {rbuf.n}')

            mct_time = time.time()

            if rbuf.is_ready:
                if verbose > 0:
                    print('Fitting actor and critic...')

                for e in range(epochs):
                    if verbose > 0:
                        print(f'Actor epoch {e}')
                    states, dists = rbuf.get_batch(target='actor')
                    self.actor.model.fit(states, dists, verbose=verbose)

                    if verbose > 0:
                        print(f'Critic epoch {e}')

                    states, values = rbuf.get_batch(target='critic')
                    self.critic.model.fit(states, values, verbose=verbose)

                if verbose > 1:
                    state, dists = rbuf.get_sample('actor')
                    print(f'Actor sample: Target={dists}, Predict={lite_actor_model.predict_single(state)}')

                    state, value = rbuf.get_sample('critic')
                    print(f'Critic sample: Target={value}, Predict={float(lite_critic_model.predict_single(state))}')

            end_time = time.time()
            if verbose > 1:
                print('MCTS time: ', datetime.timedelta(seconds=mct_time - start_time))
                print('NN time: ', datetime.timedelta(seconds=end_time - mct_time))
                print('Total time: ', datetime.timedelta(seconds=end_time - start_time))

        self.actor.checkpoint(f'{self.environment}_actor_{n_games}')
