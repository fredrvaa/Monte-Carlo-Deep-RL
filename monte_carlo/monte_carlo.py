from typing import Optional

import numpy as np

from environments.environment import State, Environment


class Node:
    def __init__(self, state: State, exploration_constant: float = 1.0, parent: Optional['Node'] = None):
        self.N: int = 0
        self.total_value: float = 0
    
        self.state: State = state
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


class MonteCarloTree:
    def __init__(self, environment: Environment, exploration_constant: float = 1.0, M: int = 100):
        self.environment: Environment = environment
        self.root: Optional[State] = None

        self.exploration_constant: float = exploration_constant
        self.M: int = M

    def _tree_search(self) -> Node:
        node = self.root
        while not node.is_leaf:
            if node.state.player == self.root.state.player:
                a = np.argmax([child.Q + child.u for child in node.children])
            else:
                a = np.argmin([child.Q - child.u for child in node.children])
            node = node.children[a]
        return node

    def _node_expansion(self, node: Node) -> None:
        successor_states = self.environment.get_successor_states(node.state)
        node.children = [Node(state=state, exploration_constant=self.exploration_constant, parent=node)
                         for state in successor_states]

    def _rollout(self, node: Node) -> float:
        state = node.state
        while not self.environment.is_final(state):
            state = self.environment.get_random_successor_state(state)

        winning_player = self.environment.winning_player(state)
        return 1.0 if winning_player == self.root.state.player else -1.0

    def _backpropagation(self, node: Node, value: float) -> None:
        while True:
            node.N += 1
            node.total_value += value
            if node.parent is None:
                break
            else:
                node = node.parent

    def simulation(self, state: State) -> list[float]:
        self.root = Node(state=state)
        for m in range(self.M):
            node = self._tree_search()
            self._node_expansion(node=node)
            if len(node.children) > 0:
                rollout_node = np.random.choice(node.children)
            else:
                rollout_node = node
            value = self._rollout(node=rollout_node)
            self._backpropagation(node=rollout_node, value=value)

        child_values = [child.Q + child.u for child in self.root.children]
        probs = [value / sum(child_values) for value in child_values]

        return probs
