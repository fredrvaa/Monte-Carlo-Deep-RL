import numpy as np
import matplotlib.pyplot as plt

from environments.environment import Environment, Player
# from environments.nim import Nim
from environments.hex import Hex
from monte_carlo.monte_carlo import MonteCarloTree

k = 4
environment = Hex(k=k)
environment.initialize(starting_player=Player.one)
mc = MonteCarloTree(environment=environment, M=400, exploration_constant=1.0)

turn = 2
final, winning_player = environment.is_final(environment.state)
while not final:
    print(f'Turn {turn // 2} | Player {Environment.get_player(environment.state)} | State: {environment.state}')
    action_probabilities = mc.simulation(environment.state)
    action = max(action_probabilities, key=lambda key: action_probabilities[key])
    print(f'Taking action {action} with probability: {action_probabilities[action]}')
    environment.step(action)
    turn += 1
    final, winning_player = environment.is_final(environment.state)

print(f'Winner: Player {winning_player}')
plt.show(block=True)
