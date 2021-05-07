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
            self.METRIC_TARGET_LENGTHS = [5, 10, 19, 24]  # @ 60 fps, in ms: 83.3, 166.7, 316.7, 400

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
        parser.add_argument('--data_workers', type=int, default=4, help='Number of parallel threads for data loading.')
        parser.add_argument('--print_every', type=int, default=200, help='Print stats to console every so many iters.')
        parser.add_argument('--eval_every', type=int, default=400, help='Evaluate validation set every so many iters.')
        parser.add_argument('--tag', default='FF', help='A custom tag for this experiment.')
        parser.add_argument('--seed', type=int, default=None, help='Random number generator seed.')

        # Data.
        parser.add_argument('--seed_seq_len', type=int, default=120, help='Number of frames for the seed length.')
        parser.add_argument('--target_seq_len', type=int, default=24, help='How many frames to predict.')
        parser.add_argument('--pose_size', type=int, default=45)
        # Learning configurations.
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs.')
        parser.add_argument('--bs_train', type=int, default=16, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=16, help='Batch size for valid/test set.')

        #model
        parser.add_argument('--hidden_size', type=int, default=1024)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.5)
        
        
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
