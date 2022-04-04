from typing import Optional
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, batch_size: int = 32, buffer_size: Optional[int] = None):
        self._x: deque[np.ndarray] = deque([], maxlen=buffer_size)
        self._y: deque[np.ndarray] = deque([], maxlen=buffer_size)
        self._n: int = 0
        self.batch_size: int = batch_size
        self.buffer_size: int = buffer_size

    def store_replay(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x.append(x)
        self._y.append(y)
        self._n += 1

    @property
    def x(self):
        return np.array(self._x)

    @property
    def y(self):
        return np.array(self._y)

    @property
    def n(self):
        return min(self._n, len(self._x))

    @property
    def is_ready(self) -> bool:
        return self.n >= self.batch_size

    def get_batch(self, replace: bool = False) -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.choice(self.n, size=self.batch_size, replace=replace)
        return np.array(self._x)[idxs], np.array(self._y)[idxs]

    def get_sample(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(self.n)
        return np.array(self._x[idx]), np.array(self._y[idx])
