from time import sleep
from typing import Optional

import numpy as np

from environments.environment import Environment, Player


class Nim(Environment):
    def __init__(self, starting_stones: int = 10, min_take: int = 1, max_take: int = 3):
        self.starting_stones: int = starting_stones
        self.binary_length: int = len(format(starting_stones, 'b'))
        self.min_take: int = min_take
        self.max_take: int = max_take
        self.n_actions: int = max_take - min_take + 1
        self.state_size: int = self.binary_length + 2  # +2 for player

    def _to_binary(self, value: int) -> np.ndarray:
        return np.array(list(format(value, f'0{self.binary_length}b')), dtype=int)

    def _to_value(self, state: np.ndarray) -> np.ndarray:
        return state[2:].dot(np.flip(2**np.arange(state[2:].shape[-1])))

    def initialize(self, starting_player: Player = Player.one) -> np.ndarray:
        state = np.concatenate((np.array(starting_player.value), self._to_binary(self.starting_stones)))
        return state

    def is_legal(self, state: np.ndarray, action: int) -> bool:
        take = action + self.min_take
        return self.min_take <= take <= self.max_take and take <= self._to_value(state)

    def is_final(self, state: np.ndarray) -> tuple[bool, Optional[Player]]:
        if self._to_value(state) == 0:
            return True, self.get_other_player(state)
        else:
            return False, None

    def perform_action(self, state: np.ndarray, action: int) -> np.ndarray:
        take = action + self.min_take
        if not self.is_legal(state, action):
            raise ValueError(f'Action {action} is not legal. Can not take {take} stones.')
        new_state = np.copy(state)
        new_state[2:] = self._to_binary(self._to_value(new_state) - take)
        self.switch_player(new_state)
        return new_state

    def visualize(self, state: np.ndarray, vis_delay: float = 0.0, vis_id: int = 1) -> None:
        print(f'{self.get_player(state)} | Stones: {self._to_value(state)}')
        sleep(vis_delay)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}_{self.starting_stones}'

