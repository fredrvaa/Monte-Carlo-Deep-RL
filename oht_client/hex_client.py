from typing import Optional

import numpy as np

from environments.hex import Hex
from learner.lite_model import LiteModel
from oht_client.ActorClient import ActorClient


class HexClient(ActorClient):
    def __init__(self, model_path: str, k: int = 7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = LiteModel.from_keras_file(model_path)

        self.k = k
        self.environment = Hex(k)

        self.series_id: Optional[int] = None

    def _num_to_binary(self, num: int):
        b = np.zeros(2, dtype=int)
        if num > 0:
            b[num-1] = 1
        return b

    def _convert_state(self, state):
        return np.concatenate([self._num_to_binary(int(num)) for num in state])

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        super().handle_series_start(unique_id, series_id, player_map, num_games, game_params)
        self.series_id = series_id

    def handle_game_over(self, winner, end_state):
        super().handle_game_over(winner, end_state)
        print("Winner!") if self.series_id == winner else print("Loser!")

    def handle_get_action(self, state):
        state = self._convert_state(state)
        dist = self.model.predict_single(state)
        action = int(self.environment.get_action_from_distribution(state, dist))
        return divmod(action, self.k)

if __name__ == '__main__':
    client = HexClient(model_path='models/Hex_7x7_actor_45.h5')
    client.run()