"""
Contains Node class used in MCT.
"""

from typing import Optional
import numpy as np


class Node:
    """
    Utility class to build monte carlo tree.
    """

    def __init__(self,
                 state: np.ndarray,
                 action: Optional[int] = None,
                 exploration_constant: float = 1.0,
                 parent: Optional['Node'] = None):
        """
        :param state: State to be stored in node
        :param action: Action taken to get to node
        :param exploration_constant: How much to weigh exploration
        :param parent: Parent node
        """
        self.N: int = 0
        self.total_value: float = 0

        self.state: np.ndarray = state
        self.action: Optional[int] = action
        self.exploration_constant = exploration_constant

        self.parent: 'Node' = parent
        self.children: list['Node'] = []

    @property
    def Q(self):
        """
        Calculates and returns Q value of node

        :return: Q value
        """

        return self.total_value / self.N

    @property
    def u(self):
        """
        Calculates and returns exploration term

        :return: Exploration term
        """
        return self.exploration_constant * np.sqrt(np.log(self.parent.N) / self.N)

    @property
    def is_leaf(self):
        """
        Whether the node is a leaf

        :return: Whether the node is a leaf
        """
        return len(self.children) == 0

    def __str__(self):
        return f'State: {self.state}, Visits: {self.N}, Value: {self.total_value}'
