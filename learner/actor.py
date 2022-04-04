from typing import Optional

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from learner.lite_model import LiteModel


def get_activation_layer(activation_string: str) -> tfkl.Layer:
    if activation_string in ['linear', 'relu', 'sigmoid', 'tanh']:
        return tfkl.Activation(activation=activation_string)
    elif activation_string == 'leaky_relu':
        return tfkl.LeakyReLU()
    else:
        raise ValueError(f'Activation {activation_string} is not available')


def get_optimizer(optimizer_string: str, **kwargs) -> tfk.optimizers.Optimizer:
    if optimizer_string == 'adam':
        return tfk.optimizers.Adam(**kwargs)
    elif optimizer_string == 'sgd':
        return tfk.optimizers.SGD(**kwargs)
    elif optimizer_string == 'adagrad':
        return tfk.optimizers.Adagrad(**kwargs)
    elif optimizer_string == 'rmsprop':
        return tfk.optimizers.RMSprop(**kwargs)
    else:
        raise ValueError(f'Optimizer {optimizer_string} is not available')


class Actor:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: list[int],
                 activation: str = 'leaky_relu',
                 optimizer: str = 'adam',
                 learning_rate: float = 5e-4,
                 decay: float = 1e-6,
                 checkpoint_folder: Optional[str] = 'models'):
        initializer = tfk.initializers.GlorotUniform()

        optimizer = get_optimizer(optimizer, learning_rate=learning_rate, decay=decay)

        self.model = tfk.Sequential()
        self.model.add(tfkl.InputLayer(input_shape=input_size))
        for size in hidden_sizes:
            self.model.add(tfkl.Dense(units=size, kernel_initializer=initializer))
            self.model.add(get_activation_layer(activation))
        self.model.add(tfkl.Dense(units=output_size, kernel_initializer=initializer))
        self.model.add(get_activation_layer('leaky_relu'))
        self.model.add(tfkl.Softmax())

        self.model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=optimizer)
        self.model.summary()

        self.checkpoint_folder: Optional[str] = checkpoint_folder

    def checkpoint(self, save_name: str):
        if self.checkpoint_folder is not None:
            tfk.models.save_model(self.model, f'{self.checkpoint_folder}/{save_name}.h5')

    def get_lite_model(self):
        return LiteModel.from_keras_model(self.model)
