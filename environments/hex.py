from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from environments.environment import Environment, Player
from environments.node_search import Node, path_to_goal_exists
from environments.utils import rotate


class Hex(Environment):
    def __init__(self, k: int = 4):
        super().__init__()

        self.k: int = k
        self.n_actions: int = k**2

        self.player1_plot, = plt.plot([], [], color='red', zorder=2, marker='o', markersize=10, ls="")
        self.player2_plot, = plt.plot([], [], color='black', zorder=2, marker='o', markersize=10, ls="")

    def initialize(self, starting_player: Player = Player.one) -> np.ndarray:
        self.state = np.zeros(2*(self.k**2 + 1))  # +1 for player
        self.state[:2] = starting_player.value

        board = self.state_to_board(self.state, self.k)
        origin = (self.k - 1) / 2
        plt.ion()
        for i, row in enumerate(board):
            for j, node in enumerate(row):
                for neighbour in node.neighbours:
                    node_x, node_y = rotate(node.row, node.col, origin, -135)
                    neighbour_x, neighbour_y = rotate(neighbour.row, neighbour.col, origin, -135)
                    plt.plot([node_x, neighbour_x], [node_y, neighbour_y], color='grey', zorder=1, markersize=10,
                             marker='o')

        self.player1_plot.set_xdata([])
        self.player1_plot.set_ydata([])

        self.player2_plot.set_xdata([])
        self.player2_plot.set_ydata([])

        plt.pause(0.05)
        plt.show(block=False)
        return self.state

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
                    if path_to_goal_exists(node, player.value, goal_nodes):
                        return True, player

        return False, None

    def perform_action(self, state: np.ndarray, action: int) -> np.ndarray:
        if not self.is_legal(state, action):
            raise ValueError(f'Action {action} is not legal. Piece can not be placed.')
        idxs = self.to_idxs(action)
        state[idxs[0]:idxs[1]+1] = state[:2]
        self.switch_player(state)
        return state

    def visualize(self, state: np.ndarray) -> None:
        player1_xdata = []
        player1_ydata = []
        player2_xdata = []
        player2_ydata = []

        board = self.state_to_board(state, self.k)
        origin = (self.k-1) / 2
        for i, row in enumerate(board):
            for j, node in enumerate(row):
                x, y = rotate(i, j, origin, -135)
                if np.array_equal(node.value, Player.one.value):
                    player1_xdata.append(x)
                    player1_ydata.append(y)
                elif np.array_equal(node.value, Player.two.value):
                    player2_xdata.append(x)
                    player2_ydata.append(y)

        self.player1_plot.set_xdata(player1_xdata)
        self.player1_plot.set_ydata(player1_ydata)
        self.player2_plot.set_xdata(player2_xdata)
        self.player2_plot.set_ydata(player2_ydata)

        plt.pause(0.05)



if __name__ == '__main__':
    hex = Hex()
    hex.initialize()
    hex.step(0)
    hex.step(1)
    hex.step(2)
    hex.step(3)
    hex.step(4)
    hex.step(5)
    hex.step(8)
    hex.step(7)
    hex.step(12)
    print(hex.state)
    print(hex.is_final(hex.state))
    hex.visualize(hex.state, hex.k)