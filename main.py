import numpy as np

from environments.nim import Nim
from monte_carlo.monte_carlo import MonteCarloTree


environment = Nim(starting_stones=13, max_take=4)
environment.initialize(starting_player=1)
mc = MonteCarloTree(environment=environment, M=500, exploration_constant=1.0)

turn = 2
while not environment.is_final():
    print(f'Turn {turn // 2} | Player {environment.state.player} | State: {environment.state.value}')
    probs = mc.simulation(environment.state)
    print(f'Action probabilities: {probs}')
    environment.step(np.argmax(probs))
    turn += 1

print(f'Winner: Player {environment.winning_player()}')