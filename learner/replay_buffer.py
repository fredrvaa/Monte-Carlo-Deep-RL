"""
Contains replay buffer class used during RL with MCT.
"""

from typing import Optional
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Utility class used to maintain a replay buffer.
    """

    def __init__(self, batch_size: int = 32, buffer_size: Optional[int] = None):
        """
        :param batch_size: Batch size retrieved from the replay buffer to be used to train the actor
        :param buffer_size: Max buffer size. If None, the buffer size is unlimited
        """

        self._x: deque[np.ndarray] = deque([], maxlen=buffer_size)
        self._y: deque[np.ndarray] = deque([], maxlen=buffer_size)
        self._n: int = 0
        self.batch_size: int = batch_size
        self.buffer_size: int = buffer_size

    def store_replay(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Stores a (observation, classification) / (x, y) pair to the replay buffer
        :param x: Observation/data
        :param y: Classification/label
        """

        self._x.append(x)
        self._y.append(y)
        self._n += 1

    @property
    def x(self):
        """
        Converts and returns stored observations (x) as np.array
        :return: np.array of all observations (x) in the replay buffer
        """

        return np.array(self._x)

    @property
    def y(self):
        """
        Converts and returns stored classifications (y) as np.array
        :return: np.array of all classifications (y) in the replay buffer
        """

        return np.array(self._y)

    @property
    def n(self):
        """
        Returns current size of replay buffer.
        :return: Current size of replay buffer
        """

        return min(self._n, len(self._x))

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the replay buffer is ready.
        The replay buffer is ready if there are more (x,y) pairs in the replay buffer than the provided batch size.

        :return: Whether the replay buffer is ready
        """

        return self.n >= self.batch_size

    def get_batch(self, replace: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Retieves a random batch from the replay buffer.

        :param replace: Whether an item can be retrieved multiple times
        :return: A random batch from the replay buffer
        """

        idxs = np.random.choice(self.n, size=self.batch_size, replace=replace)
        return np.array(self._x)[idxs], np.array(self._y)[idxs]

    def get_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retieves a random sample from the replay buffer.

        :return: A random sample from the replay buffer
        """

        idx = np.random.randint(self.n)
        return np.array(self._x[idx]), np.array(self._y[idx])
