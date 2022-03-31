from enum import Enum

import numpy as np

from abc import ABC, abstractmethod


class Player(Enum):
    one = [1, 0]
    two = [0, 1]


class Environment(ABC):
    @abstractmethod
    def initialize(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_successor_states(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_random_successor_state(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def is_legal(self, state: np.ndarray, action: int) -> bool:
        pass

    @abstractmethod
    def is_final(self, state: np.ndarray) -> tuple[bool, Player]:
        pass

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
