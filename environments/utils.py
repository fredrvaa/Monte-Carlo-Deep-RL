import numpy as np


def rotate(x: float, y: float, origin: float, angle: float) -> tuple[float, float]:
    new_x = origin + np.cos(np.radians(angle)) * (x - origin) - np.sin(np.radians(angle)) * (y - origin)
    new_y = origin + np.sin(np.radians(angle)) * (x - origin) + np.cos(np.radians(angle)) * (y - origin)
    return new_x, new_y