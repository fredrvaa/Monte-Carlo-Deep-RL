from typing import Optional

import numpy as np


class Node:
    def __init__(self, value: np.ndarray):
        self.value: np.ndarray = value
        self.row: Optional[int] = None
        self.col: Optional[int] = None
        self.neighbours: list['Node'] = []
        self.visited: bool = False


def path_to_goal_exists(node: Node, value: np.ndarray, goal_nodes: list[Node]) -> bool:
    node.visited = True
    if node in goal_nodes:
        return True

    for neighbour in node.neighbours:
        if not neighbour.visited and np.array_equal(neighbour.value, value):
            if path_to_goal_exists(neighbour, value, goal_nodes):
                return True

    return False
