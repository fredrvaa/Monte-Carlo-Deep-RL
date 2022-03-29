from typing import Optional

import numpy as np

from environments.environment import Environment, State


class Hex(Environment):
    def __init__(self, k: int = 4):
        self.k: int = k
        self.board: np.ndarray = np.zeros((k, k))
        self.state: Optional[State] = None
        self.n_actions: int = k**2

    @staticmethod
    def _bits_to_player(bits: np.ndarray) -> int:
        if bits == [0, 1]:
            return 1
        elif bits == [1, 0]:
            return 2
        else:
            return 0

    @staticmethod
    def _player_to_bits(player: int) -> list[int]:
        if player == 1:
            return [0, 1]
        elif player == 2:
            return [1, 0]
        else:
            return [0, 0]

    def _state_to_board(self, state: State) -> np.ndarray:
        board = np.reshape(
            [Hex._bits_to_player([x, y]) for x, y in zip(state.value[2::2], state.value[3::2])],
            (self.k, self.k)
        )
        return board

    def _other_player(self, player: int):
        return 1 if player == 2 else 2

    def _board_to_state(self, board: np.ndarray, player: int) -> State:
        value = np.reshape([self._player_to_bits(x) for x in np.insert(board.flatten(), 0, player)], 2+2*self.k**2)
        return State(value=value, player=player)

    def _place_piece(self, board: np.ndarray, position: np.ndarray, player: int):
        print(position)
        board[position[0]][position[1]] = player
        return board

    def initialize(self, starting_player: int = 1) -> State:
        self.board = np.zeros((self.k, self.k))

        player = [0, 1] if starting_player == 1 else [1, 0]
        initial_value = np.array(player + [0, 0] * (self.k ** 2))

        self.state = State(value=initial_value, player=starting_player)
        return self.state

    def get_successor_states(self, state: State) -> list[State]:
        board = self._state_to_board(state)
        empty = np.argwhere(board == 0)
        return [self._board_to_state(
                        self._place_piece(np.copy(board), position, state.player),
                        self._other_player(state.player)
                ) for position in empty]

    def get_random_successor_state(self, state: State) -> State:
        pass

    def winning_player(self, state: State) -> int:
        pass

    def is_legal(self, state: State, action: int) -> bool:
        return 0 <= action <= self.n_actions - 1 and state.value[2 + action*2] == 0 and state.value[3 + action*2] == 0

    def is_final(self, state: State) -> bool:
        return False

    def step(self, action: int) -> tuple[bool, State]:
        if self.state is None:
            raise ValueError("Call initialize() before trying to step")
        if not self.is_legal(self.state, action):
            raise ValueError("Action is illegal")

        if self.state.player == 1:
            self.state.value[3 + action*2] = 1
        else:
            self.state.value[2 + action*2] = 1

        self.state.player = self._other_player(self.state.player)

        return self.is_final(self.state), self.state

if __name__ == '__main__':
    hex = Hex()
    hex.initialize()
    hex.step(1)
    hex.step(3)
    print(hex.state.value)
    print(hex.is_legal(hex.state, 2))