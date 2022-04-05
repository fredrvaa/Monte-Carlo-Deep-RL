import yaml

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

    def get_environment(self):
        environment = self.config['environment']
        if 'hex' in environment:
            kwargs = environment['hex']
            return Hex(**kwargs)
        elif 'nim' in environment:
            kwargs = environment['nim']
            return Nim(**kwargs)
        else:
            raise ValueError(f'Could not load environment {environment}')


class FitParser(ConfigParser):
    def __init__(self, config_path: str):
        super().__init__(config_path)

        if self.config['type'] != 'fit':
            raise ValueError(f'Config type if {self.config["type"]} not fit')

    def get_actor_kwargs(self):
        actor_kwargs = self.config['actor']
        if 'learning_rate' in actor_kwargs:
            actor_kwargs['learning_rate'] = float(actor_kwargs['learning_rate'])
        if 'decay' in actor_kwargs:
            actor_kwargs['decay'] = float(actor_kwargs['decay'])
        return actor_kwargs

    def get_fit_kwargs(self):
        return self.config['fit']


class ToppParser(ConfigParser):
    def __init__(self, config_path: str):
        super().__init__(config_path)

        if self.config['type'] != 'topp':
            raise ValueError(f'Config type if {self.config["type"]} not topp')

    def get_round_robin_kwargs(self):
        round_robin_kwargs = self.config['round_robin']
        return round_robin_kwargs
