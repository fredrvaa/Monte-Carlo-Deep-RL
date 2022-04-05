import numpy as np

from monte_carlo.node import Node
from environments.environment import Environment, Player
from learner.lite_model import LiteModel


class MonteCarloTree:
    def __init__(self,
                 environment: Environment,
                 actor: LiteModel,
                 root_state: np.ndarray,
                 exploration_constant: float = 1.0,
                 M: int = 100,
                 epsilon: float = 0.01):

        self.environment: Environment = environment
        self.actor = actor

        self.root: Node = Node(state=root_state)

        self.exploration_constant: float = exploration_constant
        self.M: int = M
        self.epsilon = epsilon

    def _tree_search(self) -> Node:
        node = self.root
        while not node.is_leaf:
            if self.environment.get_player(node.state) == Player.one:
                a = np.argmax([child.Q + child.u if child.N != 0 else np.inf for child in node.children])
            else:
                a = np.argmin([child.Q - child.u if child.N != 0 else -np.inf for child in node.children])
            node = node.children[a]
        return node

    def _node_expansion(self, node: Node) -> None:
        legal_actions = self.environment.get_legal_actions(node.state)
        successor_states = np.array([self.environment.perform_action(np.copy(node.state), action)
                                     for action in legal_actions])
        node.children = [Node(state=state, action=action, exploration_constant=self.exploration_constant, parent=node)
                         for state, action in zip(successor_states, legal_actions)]

    def _rollout(self, node: Node) -> float:
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
        while True:
            node.N += 1
            node.total_value += value
            if node.parent is None:
                break
            else:
                node = node.parent

    def set_new_root(self, action: int):
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


if __name__ == '__main__':
    from environments.nim import Nim

    environment = Nim(starting_stones=2, max_take=5)
    state = environment.initialize(starting_player=Player.two)
    mct = MonteCarloTree(environment, None, state, M=10000, epsilon=1)
    final, winning_player = False, None
    while not final:
        dist = mct.simulation()
        print('State: ', environment._to_value(state))
        print('Dist: ', dist, int(np.argmax(dist)) + 1)
        action = int(np.argmax(dist))
        final, winning_player, state = environment.step(state, action)
        mct.set_new_root(action)
