from typing import Optional
import numpy as np

from environments.environment import State, Environment


class Nim(Environment):
    def __init__(self, starting_stones: int = 10, min_take: int = 1, max_take: int = 3):
        self.starting_stones: int = starting_stones
        self.min_take: int = min_take
        self.max_take: int = max_take
        self.state: Optional[State] = None

    def initialize(self, starting_player: int = 1) -> State:
        self.state = State(value=self.starting_stones, player=starting_player)
        return self.state

    def get_successor_states(self, state: Optional[State] = None) -> list[State]:
        if state is None:
            state = self.state

        other_player = 1 if state.player == 2 else 2
        return [State(value=state.value - take, player=other_player)
                for take in range(self.min_take, self.max_take+1) if take <= state.value]

    def get_random_successor_state(self, state: Optional[State] = None) -> State:
        if state is None:
            state = self.state

        other_player = 1 if state.player == 2 else 2
        max_take = min(self.max_take, state.value)
        take = np.random.randint(self.min_take, max_take+1) if max_take > self.min_take else max_take
        return State(value=state.value - take, player=other_player)

    def is_legal(self, state: State, action: int) -> bool:
        take = action + 1
        return self.min_take <= take <= self.max_take and take <= self.state.value

    def is_final(self, state: Optional[State] = None) -> bool:
        if state is None:
            state = self.state

        return state.value == 0

    def winning_player(self, state: Optional[State] = None) -> int:
        if state is None:
            state = self.state
        if not self.is_final(state):
            raise("State is not final")

        return 1 if state.player == 2 else 2

    def step(self, action: int) -> tuple[bool, State]:
        if self.state is None:
            raise ValueError("Call initialize() before trying to step")
        if not self.is_legal(self.state, action):
            raise ValueError("Action is illegal")

        take = action + 1
        self.state.value -= take
        self.state.player = 1 if self.state.player == 2 else 2

        return self.is_final(self.state), self.state


if __name__ == '__main__':
    nim = Nim()
    print([s.value for s in nim.get_successor_states()])
    nim.step(2)
    print([s.value for s in nim.get_successor_states()])
    nim.step(3)
    print([s.value for s in nim.get_successor_states()])
    nim.step(4)