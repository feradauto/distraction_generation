import argparse
import json
import os
import pprint
import torch


class Constants(object):
    """
    This is a singleton.
    """
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.DTYPE = torch.float32
            self.DATA_DIR = os.environ['NLP_DATA']

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):
    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        # General.
        
        #####
        parser.add_argument('--MODEL', type=str, default='t5-small')
        parser.add_argument('--TRAIN_BATCH_SIZE', type=float, default=2)
        parser.add_argument('--VALID_BATCH_SIZE', type=float, default=2)
        parser.add_argument('--TRAIN_EPOCHS', type=float, default=2)
        parser.add_argument('--VAL_EPOCHS', type=float, default=1)
        parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
        parser.add_argument('--MAX_SOURCE_TEXT_LENGTH', type=float, default=900)
        parser.add_argument('--MAX_TARGET_TEXT_LENGTH', type=float, default=901)
        parser.add_argument('--MAX_ANSWER_LENGTH', type=float, default=900)
        parser.add_argument('--SEED', type=float, default=42)
        parser.add_argument('--LAMBDA', type=float, default=0.1)
        
        config, unknown = parser.parse_known_args()
        #config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)
