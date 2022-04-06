"""
Contains a general Network class.
"""

from typing import Optional

import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from learner.lite_model import LiteModel


class Network:
    """
    General network class used to create actor and critic.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: list[int],
                 activation: str = 'relu',
                 output_activation: str = 'softmax',
                 optimizer: str = 'adam',
                 loss_function: str = 'mse',
                 learning_rate: float = 5e-4,
                 decay: float = 0.0,
                 name: Optional['str'] = None,
                 checkpoint_folder: Optional[str] = 'models'):
        """
        :param input_size: Size of input layer
        :param output_size: Size of output layer
        :param hidden_sizes: Sizes of hidden layers
        :param activation: Activation function used in hidden layers
        :param output_activation: Activation function used in the output
        :param optimizer: Optimizer to use
        :param learning_rate: Starting learning rate
        :param decay: Learning rate decay factor
        :param checkpoint_folder: Folder to checkpoint model in when checkpoint() is called
        """

        initializer = tfk.initializers.GlorotUniform()

        optimizer = self.get_optimizer(optimizer, learning_rate=learning_rate, decay=decay)

        self.model = tfk.Sequential(name=name, layers=[
            tfkl.InputLayer(input_shape=input_size),
            *[tfkl.Dense(units=size, activation=activation, kernel_initializer=initializer) for size in hidden_sizes],
            tfkl.Dense(units=output_size, activation=output_activation, kernel_initializer=initializer)
        ])

        self.model.compile(loss=loss_function, optimizer=optimizer)
        self.model.summary()

        self.checkpoint_folder: Optional[str] = checkpoint_folder

    @staticmethod
    def get_optimizer(optimizer_string: str, **kwargs) -> tfk.optimizers.Optimizer:
        """
        Utility function for converting string to a tf.keras optimizer.

        **kwargs are passed to the optimizer.

        :param optimizer_string: String for a specified tf.keras optimizer
        :return: tf.keras optimizer with provided kwargs
        """

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

    def checkpoint(self, save_name: str) -> None:
        """
        Checkpoints the current model to file.

        :param save_name: Name to save model as.
        """

        if self.checkpoint_folder is not None:
            tfk.models.save_model(self.model, f'{self.checkpoint_folder}/{save_name}.h5')

    def get_lite_model(self) -> LiteModel:
        """
        Converts current model to a LiteModel and returns it
        :return: LiteModel version of the current model
        """

        return LiteModel.from_keras_model(self.model)