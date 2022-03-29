from abc import ABC, abstractmethod


class State(ABC):
    def __init__(self, value: any, player: int = 1):
        self.value: any = value
        self.player: int = player


class Environment(ABC):
    @abstractmethod
    def initialize(self) -> State:
        pass

    @abstractmethod
    def get_successor_states(self, state: State) -> list[State]:
        pass

    @abstractmethod
    def get_random_successor_state(self, state: State) -> State:
        pass

    @abstractmethod
    def winning_player(self, state: State) -> int:
        pass

    @abstractmethod
    def is_legal(self, state: State, action: int) -> bool:
        pass

    @abstractmethod
    def is_final(self, state: State) -> bool:
        pass
