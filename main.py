import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from environments.environment import Player
from environments.nim import Nim
from environments.hex import Hex
from monte_carlo.monte_carlo import MonteCarloTree

mpl.use('TkAgg')

visualize = True

environment = Hex(k=4)
#environment = Nim(starting_stones=15, max_take=3)
environment.initialize(starting_player=Player.one)
mc = MonteCarloTree(environment=environment, M=1000, exploration_constant=1.0)

turn = 2
final, winning_player = environment.is_final(environment.state)
while not final:
    action_probabilities = mc.simulation(environment.state)
    action = max(action_probabilities, key=lambda key: action_probabilities[key])
    environment.step(action, visualize=visualize)
    turn += 1
    final, winning_player = environment.is_final(environment.state)

print(f'Winner: {winning_player}')

if type(environment) == Hex:
    plt.show(block=True)
