import os

import numpy as np

from environments.environment import Environment, Player
from learner.lite_model import LiteModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
                    vis_delay: float = 0.5,
                    vis_id: int = 1,
                    ) -> Player:
        current_player = starting_player
        state = self.environment.initialize(starting_player)
        final, winning_player = self.environment.is_final(state)

        if visualize:
            self.environment.visualize(state, vis_delay, vis_id)
        while not final:
            current_model = model1 if current_player == Player.one else model2
            dist = current_model.predict_single(state)

            action = self.environment.get_action_from_distribution(state, dist, probabilistic)
            final, winning_player, state = self.environment.step(state, action)
            if visualize:
                self.environment.visualize(state, vis_delay, vis_id)
            current_player = Player.one if current_player == Player.two else Player.two

        return winning_player

    def round_robin(self,
                    n_games: int = 10,
                    probabilistic: bool = False,
                    visualize: bool = False,
                    vis_delay: float = 0.5) -> np.ndarray:

        wins = np.zeros((len(self.models), len(self.models)))
        for i, model1 in enumerate(self.models[:-1]):
            for j, model2 in enumerate(self.models[i+1:]):
                print(f'Model {i+1} vs Model {i+j+2}')
                for n in range(n_games):
                    starting_player = Player.one if n % 2 == 0 else Player.two
                    winning_player = self.competition(model1=model1,
                                                      model2=model2,
                                                      starting_player=starting_player,
                                                      probabilistic=probabilistic,
                                                      visualize=visualize,
                                                      vis_delay=vis_delay,
                                                      vis_id=n)
                    if winning_player == Player.one:
                        wins[i][i+j+1] += 1
                    else:
                        wins[i+j+1][i] += 1

        return wins
