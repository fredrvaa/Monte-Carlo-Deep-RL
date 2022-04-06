import yaml

from environments.environment import Environment
from environments.hex import Hex
from environments.nim import Nim


class ConfigParser:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.config = config

    def get_environment(self) -> Environment:
        environment = self.config['environment']
        if 'hex' in environment:
            kwargs = environment['hex']
            return Hex(**kwargs)
        elif 'nim' in environment:
            kwargs = environment['nim']
            return Nim(**kwargs)
        else:
            raise Exception(f'Could not load environment {environment}')


class FitParser(ConfigParser):
    def __init__(self, config_path: str):
        super().__init__(config_path)

        if self.config['type'] != 'fit':
            raise ValueError(f'Config type if {self.config["type"]} not fit')

    def get_network_kwargs(self, network_type: str) -> dict:
        if network_type not in self.config:
            raise KeyError(f'Could not find network type "{network_type}" in the config')

        network_kwargs = self.config[network_type]
        if 'learning_rate' in network_kwargs:
            network_kwargs['learning_rate'] = float(network_kwargs['learning_rate'])
        if 'decay' in network_kwargs:
            network_kwargs['decay'] = float(network_kwargs['decay'])
        return network_kwargs

    def get_fit_kwargs(self) -> dict:
        return self.config['fit']


class ToppParser(ConfigParser):
    def __init__(self, config_path: str):
        super().__init__(config_path)

        if self.config['type'] != 'topp':
            raise ValueError(f'Config type if {self.config["type"]} not topp')

    def get_round_robin_kwargs(self) -> dict:
        round_robin_kwargs = self.config['round_robin']
        return round_robin_kwargs
