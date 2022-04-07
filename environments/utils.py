"""
Contains utility functions used by environments.
"""

from typing import Optional

import numpy as np


class Node:
    """
    Utility class used to aid in search.
    """

    def __init__(self, value: np.ndarray):
        self.value: np.ndarray = value
        self.row: Optional[int] = None
        self.col: Optional[int] = None
        self.neighbours: list['Node'] = []
        self.visited: bool = False


def path_to_goal_exists(node: Node, value: np.ndarray, goal_nodes: list[Node]) -> tuple[bool, list[Node]]:
    """
    Finds and returns path, if it exists, from a node to one of the goal_nodes.

    Neighbours are only visited if their values match that of current node.

    :param node: Node to perform search from
    :param value: Value to compare neighbours with
    :param goal_nodes: Nodes to search to.
    :return: Tuple of whether a path exists, and the list of nodes constituting the path
    """

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
    """
    Rotates a point about an origin.

    Assumes symmetry about origin.

    :param x: x point
    :param y: y point
    :param origin: Single origin (symmetry is assumed)
    :param angle: Angle to rotate
    :return: Tuple of rotated x, y
    """

    new_x = origin + np.cos(np.radians(angle)) * (x - origin) - np.sin(np.radians(angle)) * (y - origin)
    new_y = origin + np.sin(np.radians(angle)) * (x - origin) + np.cos(np.radians(angle)) * (y - origin)
    return new_x, new_y
