from typing import Optional
import numpy as np


class Node:
    def __init__(self,
                 state: np.ndarray,
                 action: Optional[int] = None,
                 exploration_constant: float = 1.0,
                 parent: Optional['Node'] = None):
        self.N: int = 0
        self.total_value: float = 0

        self.state: np.ndarray = state
        self.action: Optional[int] = action
        self.exploration_constant = exploration_constant

        self.parent: 'Node' = parent
        self.children: list['Node'] = []

    @property
    def Q(self):
        return self.total_value / (self.N + 1)

    @property
    def u(self):
        return self.exploration_constant * np.sqrt(np.log(self.parent.N) / (1 + self.N))

    @property
    def value(self):
        return self.Q + self.u

    @property
    def is_leaf(self):
        return len(self.children) == 0