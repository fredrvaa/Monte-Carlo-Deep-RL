from enum import Enum
from typing import Optional

import numpy as np

from abc import ABC, abstractmethod


class Player(Enum):
    one = [1, 0]
    two = [0, 1]


class Environment(ABC):
    """
    Base environment class used to implement different games with same outward interface

    Classes derived from Environment must deal with binary np.arrays where the
    two first bits describes the current player. Use Player enum to get player.
    """

    def __init__(self):
        """
        Classes derived from Environment must define n_actions and state_size.
        Important to override these values in derived classes.

        Note: This is a quick hack. Should implement this more robustly.
        See: https://stackoverflow.com/questions/55481355/python-abstract-class-shall-force-derived-classes-to-initialize-variable-in-in
        """

        self.n_actions: int = -1
        self.state_size: int = -1

    @abstractmethod
    def initialize(self, starting_player: Player = Player.one) -> np.ndarray:
        """
        Returns initial state.

        :param starting_player: Player to start in environment
        :return: Initial state
        """

        raise NotImplementedError('Derived classes must implement initialize()')

    @abstractmethod
    def is_legal(self, state: np.ndarray, action: int) -> bool:
        """
        Checks if action is legal in state.

        :param state: np.array describing state
        :param action: Action to check
        :return: Whether action is legal in state
        """

        raise NotImplementedError('Derived classes must implement is_legal()')

    @abstractmethod
    def is_final(self, state: np.ndarray) -> tuple[bool, Optional[Player]]:
        """
        Checks state is final, and if it is also returns winning player.

        :param state: np.array describing state
        :return: Tuple consisting of: Whether state is final, winning player
        """

        raise NotImplementedError('Derived classes must implement is_final()')

    @abstractmethod
    def perform_action(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Performs action on state and returns resulting state.

        :param state: np.array describing state
        :param action: Action to perform
        :return: Tuple consisting of: Whether state is final, winning player
        """

        raise NotImplementedError('Derived classes must implement perform_action()')

    @abstractmethod
    def visualize(self, state: np.ndarray, vis_delay: float, vis_id: int) -> None:
        """
        Visualizes state of environment.

        :param state: np.array describing state to be visualized
        :param vis_delay: Delay after visualization
        :param vis_id: Specific visualization id. Useful if multiple games are played.
        """

        raise NotImplementedError('Derived classes must implement perform_action()')

    def get_action_from_distribution(self, state: np.ndarray, dist: np.ndarray, probabilistic: bool = False) -> int:
        """
        Retrieves a legal action based on a distribution.

        if probabilistic:
            Sample the legal distribution
        else:
            Return legal action with highest probability

        :param state: np.array describing state
        :param dist: Distribution over actions
        :param probabilistic: Whether to sample or pick highest probability action
        :return: Action based on distribtuion
        """

        legal_dist = np.array([dist[action] if self.is_legal(state, action) else 0.0 for action in range(self.n_actions)])
        dist_sum = legal_dist.sum()
        if dist_sum == 0:
            return self.get_random_action(state)
        legal_dist /= legal_dist.sum()
        return np.random.choice(np.arange(legal_dist.shape[0]), p=legal_dist) if probabilistic else np.argmax(legal_dist)

    def get_legal_actions(self, state: np.ndarray) -> np.ndarray:
        """
        Returns all legal actions in state.

        :param state: np.array describing state
        :return: All legal actions in state
        """

        return np.array([action for action in range(self.n_actions) if self.is_legal(state, action)])

    def get_successor_states(self, state: np.ndarray) -> np.ndarray:
        """
        Returns all legal successor states of a state.

        :param state: np.array describing state
        :return: All legal successor states of a state
        """

        return np.array([self.perform_action(np.copy(state), action) for action in self.get_legal_actions(state)])

    def get_random_action(self, state: np.ndarray) -> int:
        """
        Returns a random legal action in state.

        :param state: np.array describing state
        :return: A random legal action in state
        """

        return np.random.choice(self.get_legal_actions(state))

    def get_random_successor_state(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a random legal successor state of a state

        :param state: np.array describing state
        :return: A random legal successor state of a state
        """

        action = self.get_random_action(state)
        return self.perform_action(np.copy(state), action)

    def step(self, state: np.ndarray, action: int) -> tuple[bool, Optional[Player], np.ndarray]:
        """
        Performs action in state and returns the new state, along with whether state is final, and winning player.

        Action should be checked if legal before calling step!

        :param state: np.array describing state
        :param action: Action to perform in state
        :return: Tuple of whether the state is final, and potential winning player, successor state
        """

        new_state = self.perform_action(state, action)
        final, winning_player = self.is_final(new_state)

        return final, winning_player, new_state

    @staticmethod
    def get_player(state: np.ndarray) -> Player:
        """
        Gets the current player from np.array describing state.

        :param state: np.array describing state where the two first bits describe the player
        :return: A Player object corresponding to the current player
        """

        return Player(state[:2].tolist())

    @staticmethod
    def get_other_player(state: np.ndarray) -> Player:
        """
        Gets the other player from np.array describing state.

        :param state: np.array describing state where the two first bits describe the player
        :return: A Player object corresponding to the other player (not current)
        """

        return Player(np.flip(state[:2]).tolist())

    @staticmethod
    def switch_player(state: np.ndarray) -> np.ndarray:
        """
        Returns state where the bits describing the current player are flipped.

        This method should generally be used at the end of perform_action().

        :param state: np.array describing state where the two first bits describe the player
        :return: The same state, but with the two first bits flipped.
        """
        state[:2] = np.flip(state[:2])
        return state

    @abstractmethod
    def __str__(self) -> str:
        return 'Base_Environment'
