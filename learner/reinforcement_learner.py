"""
Contains the core reinforcement learning algorithm.
"""

import time
import datetime
import numpy as np
from typing import Optional

import tensorflow.keras.backend as K

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
                 ):
        """
        :param environment: Environment the actor should be fitted to
        :param actor: Actor to fit
        :param critic: Critic used to evaluate states during MCTS
        """

        self.environment: Environment = environment
        self.actor: Network = actor
        self.critic: Network = critic

    def _decay_param(self, param: float, decay: float) -> float:
        return param * (1 / (1 + decay))

    def fit(self,
            n_games: int = 100,
            n_search_games: int = 500,
            batch_size: int = 20,
            buffer_size: Optional[int] = None,
            epochs: int = 1,
            critic_discount: float = 0.02,
            epsilon: float = 1.0,
            sigma: float = 1.0,
            epsilon_decay: float = 0.0,
            sigma_decay: float = 0.0,
            n_saved_models: int = 2,
            visualize: bool = False,
            vis_delay: float = 0.1,
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
        :param critic_discount: Discount factor for sequential states in a single game
        :param epsilon: Probability of taking random action instead of actor during MCTS rollout
        :param sigma: Probability of using rollout in MCTS instead of critic evaluation
        :param epsilon_decay: Decay factor for epsilon. 0.0 -> No decay
        :param sigma_decay: Decay factor for sigma. 0.0 -> No decay
        :param visualize: Whether to visualize each game
        :param vis_delay: Delay between each visualization
        :param n_saved_models: Number of models to save to file (checkpoint)
        :param verbose: Decides verbosity during fit: 0=minimal, 1=essential, 2=additional, 3=predictions per episode
        """

        checkpoint_iter = n_games // (n_saved_models - 1)

        rbuf = ReplayBuffer(targets=['actor', 'critic'] if self.critic is not None else ['actor'],
                            batch_size=batch_size,
                            buffer_size=buffer_size)

        start_train = False
        for n in range(n_games):
            # Checkpoint model
            if checkpoint_iter is not None and n % checkpoint_iter == 0:
                model_name = f'{self.environment}_actor_{n}'
                self.actor.checkpoint(model_name)
                print(f'Checkpointed {model_name}')

            print(f'----Game {n + 1}----')
            start_time = time.time()

            # Update epsilon and sigma
            if start_train:
                epsilon = self._decay_param(epsilon, epsilon_decay)
                sigma = self._decay_param(sigma, sigma_decay)

            if verbose > 1:
                print(f'Epsilon: {epsilon}')
                print(f'Actor lr: {K.eval(self.actor.model.optimizer._decayed_lr(float))}')
                if self.critic is not None:
                    print(f'Sigma: {sigma}')
                    print(f'Actor lr: {K.eval(self.critic.model.optimizer._decayed_lr(float))}')

            # Convert models to LiteModel for faster predictions
            lite_actor_model = self.actor.get_lite_model()
            lite_critic_model = self.critic.get_lite_model() if self.critic is not None else None

            # Initialize state
            state = self.environment.initialize(np.random.choice(list(Player)))
            if visualize:
                self.environment.visualize(state, vis_delay, n)

            # Used to store training cases for actor/critic
            visited_states = []
            targets = {'actor': []}

            # Initialize MCT and run MCTS
            mct = MonteCarloTree(environment=self.environment,
                                 actor=lite_actor_model,
                                 critic=lite_critic_model,
                                 root_state=state,
                                 M=n_search_games)

            final, winning_player = False, None

            if verbose > 0:
                print('Building replay buffer...')

            while not final:
                dist = mct.simulation(epsilon=epsilon, sigma=sigma)

                visited_states.append(state)
                targets['actor'].append(dist)

                if np.random.random() > epsilon:
                    action = self.environment.get_action_from_distribution(state, dist)
                else:
                    action = self.environment.get_random_action(state)

                final, winning_player, state = self.environment.step(state, action)
                if visualize:
                    self.environment.visualize(state, vis_delay, n)
                mct.set_new_root(action)

            # Add critic targets if critic is used
            if self.critic is not None:
                n_states = len(visited_states)
                critic_targets = [(1.0-critic_discount)**(n_states - (i + 1)) for i in range(len(visited_states))]
                if winning_player == Player.two:
                    critic_targets = [-t for t in critic_targets]
                targets['critic'] = critic_targets

            # Store replays in buffer
            rbuf.store_replays(states=visited_states, **targets)

            if verbose > 1:
                print(f'Buffer size: {rbuf.n}')

            mct_time = time.time()

            # Train NNs if replay buffer is larger than batch size
            if rbuf.is_ready:
                start_train = True
                if verbose > 0:
                    print('Fitting nerual networks...')

                for e in range(epochs):
                    if verbose > 0:
                        print(f'Epoch {e}')

                    states, dists = rbuf.get_batch(target='actor')
                    loss = self.actor.model.train_on_batch(states, dists)
                    if verbose > 0:
                        print(f'Actor loss: {loss}')

                    if self.critic is not None:
                        states, values = rbuf.get_batch(target='critic')
                        loss = self.critic.model.train_on_batch(states, values)
                        if verbose > 0:
                            print(f'Critic loss: {loss}')

                    if verbose > 2:
                        state, dists = rbuf.get_sample('actor')
                        print(f'Actor sample: Target={dists}, Predict={lite_actor_model.predict_single(state)}')

                        if self.critic is not None:
                            state, value = rbuf.get_sample('critic')
                            print(
                                f'Critic sample: Target={value}, Predict={float(lite_critic_model.predict_single(state))}')

            end_time = time.time()
            if verbose > 1:
                print('MCTS time: ', datetime.timedelta(seconds=mct_time - start_time))
                print('NN time: ', datetime.timedelta(seconds=end_time - mct_time))
                print('Total time: ', datetime.timedelta(seconds=end_time - start_time))

        self.actor.checkpoint(f'{self.environment}_actor_{n_games}')
