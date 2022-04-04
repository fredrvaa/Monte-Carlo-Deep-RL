import numpy as np


class ReplayBuffer:
    def __init__(self, batch_size: int = 32):
        self._x = []
        self._y = []
        self.n = 0
        self.batch_size = batch_size

    def store_replay(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x.append(x)
        self._y.append(y)
        self.n += 1

    @property
    def is_ready(self) -> bool:
        return self.n >= self.batch_size

    def get_batch(self, replace: bool = False) -> tuple[np.ndarray, np.ndarray]:
        idxs = np.random.choice(self.n, size=self.batch_size, replace=replace)
        return np.array(self._x)[idxs], np.array(self._y)[idxs]

    def get_sample(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(self.n)
        return np.array(self._x[idx]), np.array(self._y[idx])
