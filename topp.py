import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from environments.environment import Environment, Player
from environments.hex import Hex
from environments.nim import Nim
from learner.lite_model import LiteModel


class Topp:
    def __init__(self, environment: Environment, model_paths: list[str]):
        self.environment = environment
        self.models: list[LiteModel] = [LiteModel.from_keras_file(path) for path in model_paths]

    def competition(self,
                    model1: LiteModel,
                    model2: LiteModel,
                    starting_player: Player = Player.one,
                    probabilistic: bool = False,
                    visualize: bool = False,
                    ) -> Player:
        current_player = starting_player
        state = self.environment.initialize(starting_player)
        final, winning_player = self.environment.is_final(state)

        while not final:
            current_model = model1 if current_player == Player.one else model2
            dist = current_model.predict_single(state)
            action = self.environment.get_action_from_distribution(state, dist, probabilistic)

            final, winning_player, state = self.environment.step(state, action, visualize)
            current_player = Player.one if current_player == Player.two else Player.two

        return winning_player

    def round_robin(self, n_games: int = 10, probabilistic: bool = False) -> np.ndarray:
        wins = np.zeros(len(self.models))
        for i, model1 in enumerate(self.models[:-1]):
            for j, model2 in enumerate(self.models[i+1:]):
                print(f'Model {i+1} vs Model {i+j+2}')
                for n in range(n_games):
                    starting_player = Player.one if n % 2 == 0 else Player.two
                    winning_player = self.competition(model1, model2, starting_player, probabilistic)
                    if winning_player == Player.one:
                        wins[i] += 1
                    else:
                        wins[i+j+1] += 1
        return wins


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help='path/to/models/folder', default='models')
    parser.add_argument('-p', '--probabilistic', action='store_true', help='Whether actions should be sampled')
    args = parser.parse_args()

    environment = Nim(starting_stones=15, max_take=5)

    model_paths = [f'{args.folder}/{model_name}' for model_name in os.listdir(args.folder)]
    model_paths.sort(key=lambda p: int(p.split('.')[0].split('_')[-1]))
    print(model_paths)
    topp = Topp(environment, model_paths)

    wins = topp.round_robin(n_games=30, probabilistic=args.probabilistic)
    print(wins)

