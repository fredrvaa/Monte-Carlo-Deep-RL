from typing import Optional

import numpy as np


class Node:
    def __init__(self, value: np.ndarray):
        self.value: np.ndarray = value
        self.row: Optional[int] = None
        self.col: Optional[int] = None
        self.neighbours: list['Node'] = []
        self.visited: bool = False


def path_to_goal_exists(node: Node, value: np.ndarray, goal_nodes: list[Node]) -> tuple[bool, list[Node]]:
    node.visited = True
    if node in goal_nodes:
        return True, [node]

    for neighbour in node.neighbours:
        if not neighbour.visited and np.array_equal(neighbour.value, value):
            exists, path = path_to_goal_exists(neighbour, value, goal_nodes)
            if exists:
                path.append(node)
                return True, path

    return False, []


def rotate(x: float, y: float, origin: float, angle: float) -> tuple[float, float]:
    new_x = origin + np.cos(np.radians(angle)) * (x - origin) - np.sin(np.radians(angle)) * (y - origin)
    new_y = origin + np.sin(np.radians(angle)) * (x - origin) + np.cos(np.radians(angle)) * (y - origin)
    return new_x, new_y