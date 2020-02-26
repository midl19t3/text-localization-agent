import os
import json
from configparser import ConfigParser
from argparse import ArgumentParser
from agent.utils import ensure_folder


CONFIG = dict()

def load_config(path=None):
    _configparser = ConfigParser()
    _argparser = ArgumentParser()
    _argparser.add_argument('--config', help='Path to config file')
    args, _ = _argparser.parse_known_args()

    if args.config:
        if not os.path.exists(args.config):
            raise Exception('Configuration file {} does not exist'.format(args.config))
        _configparser.read(args.config)
    elif path:
        _configparser.read(path)

    # == dataset ==
    CONFIG['dataset'] = _configparser.get('dataset', 'dataset')
    CONFIG['dataset_path'] = _configparser.get('dataset', 'dataset_path')

    # == experiment ==
    CONFIG['experiment_id'] = _configparser.get('experiment', 'experiment_id', fallback='unnamed')
    CONFIG['gpu_id'] = _configparser.getint('experiment', 'gpu_id', fallback=-1)
    CONFIG['seed_agent'] = _configparser.getint('experiment', 'seed_agent', fallback=20301)
    CONFIG['seed_environment'] = _configparser.getint('experiment', 'seed_environment', fallback=20302)

    # == training ==
    CONFIG['steps'] = _configparser.getint('training', 'steps', fallback=5000)
    CONFIG['train_max_episode_len'] = _configparser.getint('training', 'train_max_episode_len', fallback=100)

    # optimizer
    CONFIG['epsilon'] = _configparser.getfloat('training', 'epsilon', fallback=0.01)
    CONFIG['learning_rate'] = _configparser.getfloat('training', 'learning_rate', fallback=0.0001)

    # agent
    CONFIG['gamma'] = _configparser.getfloat('training', 'gamma', fallback=0.1)
    CONFIG['replay_start_size'] = _configparser.getint('training', 'replay_start_size', fallback=100)
    CONFIG['replay_buffer_capacity'] = _configparser.getint('training', 'replay_buffer_capacity', fallback=1000)
    CONFIG['update_interval'] = _configparser.getint('training', 'update_interval', fallback=1)
    CONFIG['target_update_interval'] = _configparser.getint('training', 'target_update_interval', fallback=100)

    # explorer
    CONFIG['start_epsilon'] = _configparser.getfloat('training', 'start_epsilon', fallback=1.0)
    CONFIG['end_epsilon'] = _configparser.getfloat('training', 'end_epsilon', fallback=0.1)
    CONFIG['decay_steps'] = _configparser.getint('training', 'decay_steps', fallback=300000)

    # == evaluation ==
    CONFIG['eval_n_episodes'] = _configparser.getint('evaluation', 'eval_n_episodes', fallback=10)
    CONFIG['eval_interval'] = _configparser.getint('evaluation', 'eval_interval', fallback=500)
    CONFIG['use_tensorboard'] = _configparser.getboolean('evaluation', 'use_tensorboard', fallback=False)
    CONFIG['tensorboard'] = _configparser.getboolean('evaluation', 'use_tensorboard', fallback=False)

    # == environment ==

    # Train only
    CONFIG['playout_episode'] = _configparser.getboolean('environment', 'playout_episode', fallback=False)
    CONFIG['premasking'] = _configparser.getboolean('environment', 'premasking', fallback=True)

    # Train + Test
    CONFIG['max_steps_per_image'] = _configparser.getint('environment', 'max_steps_per_image', fallback=200)
    CONFIG['bbox_scaling'] = _configparser.getfloat('environment', 'bbox_scaling', fallback=.0)
    CONFIG['bbox_transformer'] = _configparser.get('environment', 'bbox_transformer', fallback='base')
    CONFIG['ior_marker_type'] = _configparser.get('environment', 'ior_marker_type', fallback='cross')
    CONFIG['has_termination_action'] = _configparser.getboolean('environment', 'has_termination_action', fallback=False)

    # if set, override config w/ command line arguments
    for key in CONFIG:
        _argparser.add_argument('--{}'.format(key), type=type(CONFIG[key]))
        args, _ = _argparser.parse_known_args()
        override =  vars(args)[key]
        if override:
            CONFIG[key] = override

    return CONFIG


def write_config(path):
    cfg = ConfigParser()
    cfg.read_dict({'agent': CONFIG})
    ensure_folder(path)

    with open(os.path.join(path, 'config.ini'), 'w') as f:
        cfg.write(f)

    print('Saved configuration file to {}'.format(path))


def print_config():
    print('Running w/ config:\n' + json.dumps(CONFIG, indent=4))


load_config()
