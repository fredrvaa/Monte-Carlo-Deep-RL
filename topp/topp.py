"""
Contains class for running a Tournament of Progressive Policies (TOPP).
"""

import os

import numpy as np

from environments.environment import Environment, Player
from learner.lite_model import LiteModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Topp:
    """
    Class used to run a Tournament of Progressive Policies (TOPP).
    """

    def __init__(self, environment: Environment, model_paths: list[str]):
        """

        :param environment: Environment to run the TOPP in. Must correspond with the trained models
        :param model_paths: List of file paths to trained models. Must correspond with the environment
        """

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
        """
        Runs a single competition between two models.

        :param model1: First model to compete
        :param model2: Second model to compete
        :param starting_player: Which player should start
        :param probabilistic: Whether actions should be sampled
                              If not, the action with highest probability is always chosen
        :param visualize: Whether to visualize the competition
        :param vis_delay: Delay between each visualization
        :param vis_id: Id of the visualization. Can be used by matplotlib to distinguish between figures
        :return: The winning player
        """

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
        """
        Runs a round robin tournament with all models, and returns the number of wins for each model.

        In this tournament form, each model competes against each other model n_games times.

        The number of wins for each model is returned in the form of a 2D np.array of size MxM
        where M is the number of models. Each row in the array specifies how many times that model has won against the
        other models.

        Example:
            Game setup:
                n_games = 5
                M = 3
            Return:
                [[0, 2, 5],
                 [3, 0, 4],
                 [0, 1, 0]]
            Explanation:
                Model1 vs Model2: 2 to 3
                Model1 vs model3: 5 to 0
                Model2 vs Model3: 4 to 1

        :param n_games: Number of games each model plays against any other model
        :param probabilistic: Whether actions should be sampled
                              If not, the action with highest probability is always chosen
        :param visualize: Whether to visualize the competition
        :param vis_delay: Delay between each visualization
        :return: np.array with the number of wins. See Example for output format.
        """

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
