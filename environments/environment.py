from enum import Enum
from typing import Optional

import numpy as np

from abc import ABC, abstractmethod


class Player(Enum):
    one = [1, 0]
    two = [0, 1]


class Environment(ABC):
    def __init__(self):
        self.state: Optional[np.ndarray] = None

    @abstractmethod
    def initialize(self) -> np.ndarray:
        pass

    @abstractmethod
    def is_legal(self, state: np.ndarray, action: int) -> bool:
        pass

    @abstractmethod
    def is_final(self, state: np.ndarray) -> tuple[bool, Player]:
        pass

    @abstractmethod
    def perform_action(self, state: np.ndarray, action: int) -> np.ndarray:
        pass

    @abstractmethod
    def visualize(self, state: np.ndarray) -> None:
        pass

    def get_legal_actions(self, state: np.ndarray) -> np.ndarray:
        return np.array([action for action in range(self.n_actions) if self.is_legal(state, action)])

    def get_successor_states(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.perform_action(np.copy(state), action) for action in self.get_legal_actions(state)])

    def get_random_action(self, state: np.ndarray) -> int:
        return np.random.choice(self.get_legal_actions(state))

    def get_random_successor_state(self, state: np.ndarray) -> np.ndarray:
        action = self.get_random_action(state)
        return self.perform_action(np.copy(state), action)

    def step(self, action: int, visualize: bool = False) -> tuple[bool, Optional[Player], np.ndarray]:
        if self.state is None:
            raise ValueError('Call initialize() before trying to step')

        if visualize:
            self.visualize(self.state)

        self.perform_action(self.state, action)
        final, winning_player = self.is_final(self.state)

        if final and visualize:
            self.visualize(self.state)

        return final, winning_player, self.state

    @staticmethod
    def get_player(state: np.ndarray) -> Player:
        return Player(state[:2].tolist())

    @staticmethod
    def get_other_player(state: np.ndarray) -> Player:
        return Player(np.flip(state[:2]).tolist())

    @staticmethod
    def switch_player(state: np.ndarray) -> np.ndarray:
        state[:2] = np.flip(state[:2])
        return state
