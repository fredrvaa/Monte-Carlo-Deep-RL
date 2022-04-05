from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from environments.environment import Environment, Player
from environments.utils import Node, path_to_goal_exists, rotate


class Hex(Environment):
    def __init__(self, k: int = 4):
        self.k: int = k
        self.n_actions: int = k**2
        self.state_size: int = 2 * (self.n_actions + 1)  # +1 for player

    def initialize(self, starting_player: Player = Player.one) -> np.ndarray:
        state = np.zeros(self.state_size)
        state[:2] = starting_player.value

        return state

    @staticmethod
    def to_idxs(position: int) -> tuple[int, int]:
        idx = 2 + position * 2
        return idx, idx+1

    @staticmethod
    def to_piece(state: np.ndarray, position: int) -> np.ndarray:
        idx = 2 + position * 2
        return state[idx:idx+2]

    @staticmethod
    def state_to_board(state: np.ndarray, k: int) -> np.ndarray:
        # Construct board
        board = np.reshape([Node(Hex.to_piece(state, a)) for a in range(k ** 2)], (k, k))

        # Add neighbours
        for row in range(k):
            for col in range(k):
                board[row][col].row = row
                board[row][col].col = col
                if row > 0:
                    board[row][col].neighbours.append(board[row - 1][col])
                if row < k - 1:
                    board[row][col].neighbours.append(board[row + 1][col])
                if col > 0:
                    board[row][col].neighbours.append(board[row][col - 1])
                if col < k - 1:
                    board[row][col].neighbours.append(board[row][col + 1])
                if col < k - 1 and row > 0:
                    board[row][col].neighbours.append(board[row - 1][col + 1])
                if col > 0 and row < k - 1:
                    board[row][col].neighbours.append(board[row + 1][col - 1])

        return board

    def is_legal(self, state: np.ndarray, action: int) -> bool:
        idxs = self.to_idxs(action)
        return 0 <= action <= self.n_actions - 1 and state[idxs[0]] == 0 and state[idxs[1]] == 0

    def is_final(self, state: np.ndarray) -> tuple[bool, Optional[Player]]:
        board = self.state_to_board(state, self.k)

        for player in Player:
            if player == Player.one:
                start_nodes = board[0, :]
                goal_nodes = board[-1, :]
            else:
                start_nodes = board[:, 0]
                goal_nodes = board[:, -1]

            for node in start_nodes:
                if np.array_equal(node.value, player.value):
                    exists, path = path_to_goal_exists(node, player.value, goal_nodes)
                    if exists:
                        return True, player

        return False, None

    def perform_action(self, state: np.ndarray, action: int) -> np.ndarray:
        if not self.is_legal(state, action):
            raise ValueError(f'Action {action} is not legal. Piece can not be placed.')
        idxs = self.to_idxs(action)
        new_state = np.copy(state)
        new_state[idxs[0]:idxs[1]+1] = new_state[:2]
        self.switch_player(new_state)
        return new_state

    def visualize(self, state: np.ndarray, vis_delay: float = 0.1, vis_id: int = 1) -> None:
        # Get figure and clear
        fig = plt.figure(vis_id)
        fig.clear()

        # Plot base board
        board = self.state_to_board(state, self.k)
        origin = (self.k-1) / 2
        for i, row in enumerate(board):
            for j, node in enumerate(row):
                for neighbour in node.neighbours:
                    node_x, node_y = rotate(node.row, node.col, origin, -135)
                    neighbour_x, neighbour_y = rotate(neighbour.row, neighbour.col, origin, -135)
                    plt.plot([node_x, neighbour_x], [node_y, neighbour_y], color='grey', zorder=1, markersize=10,
                             marker='o')

        # Plot player pieces
        player1_x = []
        player1_y = []
        player2_x = []
        player2_y = []
        for i, row in enumerate(board):
            for j, node in enumerate(row):
                x, y = rotate(i, j, origin, -135)
                if np.array_equal(node.value, Player.one.value):
                    player1_x.append(x)
                    player1_y.append(y)
                elif np.array_equal(node.value, Player.two.value):
                    player2_x.append(x)
                    player2_y.append(y)

        plt.plot(player1_x, player1_y, color='red', zorder=3, marker='o', markersize=10, ls="")
        plt.plot(player2_x, player2_y, color='black', zorder=3, marker='o', markersize=10, ls="")

        # Check if state is final and if so get path
        final, winning_player, path = False, None, []
        for player in Player:
            if player == Player.one:
                start_nodes = board[0, :]
                goal_nodes = board[-1, :]
            else:
                start_nodes = board[:, 0]
                goal_nodes = board[:, -1]

            for node in start_nodes:
                if np.array_equal(node.value, player.value):
                    exists, found_path = path_to_goal_exists(node, player.value, goal_nodes)
                    if exists:
                        final, winning_player, path = True, player, found_path

        # Plot path if state is final
        if final:
            fig.suptitle(f'{winning_player} wins!', color='red' if winning_player == Player.one else 'black')
            for node1, node2 in zip(path, path[1:]):
                node1_x, node1_y = rotate(node1.row, node1.col, origin, -135)
                node2_x, node2_y = rotate(node2.row, node2.col, origin, -135)
                plt.plot([node1_x, node2_x], [node1_y, node2_y], color='orange', zorder=2)

        plt.pause(vis_delay)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}_{self.k}x{self.k}'
