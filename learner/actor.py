from typing import Optional

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from learner.lite_model import LiteModel


class Actor:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: list[int],
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 checkpoint_folder: Optional[str] = 'models'):
        initializer = tfk.initializers.GlorotUniform()

        self.model = tfk.Sequential()
        self.model.add(tfkl.InputLayer(input_shape=input_size))
        for size in hidden_sizes:
            self.model.add(tfkl.Dense(units=size, activation=activation, kernel_initializer=initializer))
        self.model.add(tfkl.Dense(units=output_size, activation=activation, kernel_initializer=initializer))
        self.model.add(tfkl.Softmax())
        optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=optimizer)
        self.model.summary()

        self.checkpoint_folder: Optional[str] = checkpoint_folder

    def checkpoint(self, save_name: str):
        if self.checkpoint_folder is not None:
            tfk.models.save_model(self.model, f'{self.checkpoint_folder}/{save_name}.h5')

    def get_lite_model(self):
        return LiteModel.from_keras_model(self.model)
