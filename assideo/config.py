import os

from omegaconf import OmegaConf

DEFAULT_CONFIG_PATH = 'configs/default_config.yaml'


def default_config():
    return OmegaConf.load(
        os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_PATH))


def load_configs(*args):
    return OmegaConf.merge(default_config(),
                           *[OmegaConf.load(path) for path in args],
                           OmegaConf.from_cli())
