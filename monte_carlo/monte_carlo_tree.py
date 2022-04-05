"""
Contains MCT class.
"""

import numpy as np

from monte_carlo.node import Node
from environments.environment import Environment, Player
from learner.lite_model import LiteModel


class MonteCarloTree:
    """
    Class used to perform monte carlo simulations.
    """

    def __init__(self,
                 environment: Environment,
                 actor: LiteModel,
                 root_state: np.ndarray,
                 exploration_constant: float = 1.0,
                 M: int = 100,
                 epsilon: float = 0.01):
        """
        :param environment: Environment to perform search
        :param actor: Actor used to take actions during rollout
        :param root_state: Root state in environment
        :param exploration_constant: How much to weigh exploration
        :param M: Number of searches in simulation
        :param epsilon: Random action is taken in rollout with probability epsilon
        """

        self.environment: Environment = environment
        self.actor = actor

        self.root: Node = Node(state=root_state)

        self.exploration_constant: float = exploration_constant
        self.M: int = M
        self.epsilon = epsilon

    def _tree_search(self) -> Node:
        """
        Searches the tree for a leaf node starting at the root using tree policy.

        :return: A leaf node
        """

        node = self.root
        while not node.is_leaf:
            if self.environment.get_player(node.state) == Player.one:
                a = np.argmax([child.Q + child.u if child.N != 0 else np.inf for child in node.children])
            else:
                a = np.argmin([child.Q - child.u if child.N != 0 else -np.inf for child in node.children])
            node = node.children[a]
        return node

    def _node_expansion(self, node: Node) -> None:
        """
        Adds children to a node.

        :param node: Node to expand
        """

        legal_actions = self.environment.get_legal_actions(node.state)
        successor_states = np.array([self.environment.perform_action(np.copy(node.state), action)
                                     for action in legal_actions])
        node.children = [Node(state=state, action=action, exploration_constant=self.exploration_constant, parent=node)
                         for state, action in zip(successor_states, legal_actions)]

    def _rollout(self, node: Node) -> float:
        """
        Performs rollout from a node to give a value of the node.

        :param node: Node to perform rollout from
        :return: Value from the single rollout
        """

        state = node.state
        final, winning_player = self.environment.is_final(state)
        while not final:
            if np.random.random() > self.epsilon:
                dist = self.actor.predict_single(state)
                action = self.environment.get_action_from_distribution(state, dist)
            else:
                action = self.environment.get_random_action(state)

            final, winning_player, state = self.environment.step(state, action)

        reward = 1.0 if winning_player == Player.one else -1.0
        return reward

    def _backpropagation(self, node: Node, value: float) -> None:
        """
        Backpropagates a value from a node up to the root.

        :param node: Node to backpropagate from
        :param value: Value to backpropagate
        """

        while True:
            node.N += 1
            node.total_value += value
            if node.parent is None:
                break
            else:
                node = node.parent

    def set_new_root(self, action: int) -> None:
        """
        Moves the root to one of its children, and discards the rest of the tree.

        :param action: Action to perform to move the root
        """

        new_root = None
        for child in self.root.children:
            if child.action == action:
                new_root = child
                break
        if new_root is None:
            raise ValueError(f'Action {action} does not move root to any legal child state.')

        new_root.parent = None
        self.root = new_root

    def simulation(self) -> np.ndarray:
        """
        Performs monte carlo simulation and returns action probabilities

        :return: Probabilities for taking different actions, where high probabilities
                 correspond to good actions in the current root state
        """
        for m in range(self.M):
            node = self._tree_search()
            self._node_expansion(node=node)
            if len(node.children) > 0:
                rollout_node = np.random.choice(node.children)
            else:
                rollout_node = node
            value = self._rollout(node=rollout_node)
            self._backpropagation(node=rollout_node, value=value)

        visit_sum = sum([child.N for child in self.root.children])
        dist = {child.action: child.N / visit_sum for child in self.root.children}
        return np.array([0 if i not in dist else dist[i] for i in range(self.environment.n_actions)])

