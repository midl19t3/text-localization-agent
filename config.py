from configparser import ConfigParser
from argparse import ArgumentParser
import os
import json


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

    # data and paths
    CONFIG['imagefile_path'] = _configparser.get('agent', 'imagefile_path', fallback='../generated_data/image_locations.txt')
    CONFIG['boxfile_path'] = _configparser.get('agent', 'boxfile_path', fallback='../generated_data/bounding_boxes.npy')
    CONFIG['resultdir_path'] = _configparser.get('agent', 'resultdir_path', fallback='./results')
    CONFIG['agentdir_path'] = _configparser.get('agent', 'agentdir_path', fallback='./agent')

    # hardware
    CONFIG['gpu_id'] = _configparser.getint('agent', 'gpu_id', fallback=-1)

    # optimizer
    CONFIG['epsilon'] = _configparser.getfloat('agent', 'epsilon', fallback=0.01)
    CONFIG['learning_rate'] = _configparser.getfloat('agent', 'learning_rate', fallback=0.005)

    # agent
    CONFIG['gamma'] = _configparser.getfloat('agent', 'gamma', fallback=0.1)
    CONFIG['replay_start_size'] = _configparser.getint('agent', 'replay_start_size', fallback=100)
    CONFIG['replay_buffer_capacity'] = _configparser.getint('agent', 'replay_buffer_capacity', fallback=1000)
    CONFIG['update_interval'] = _configparser.getint('agent', 'update_interval', fallback=1)
    CONFIG['target_update_interval'] = _configparser.getint('agent', 'target_update_interval', fallback=100)

    # explorer
    CONFIG['start_epsilon'] = _configparser.getfloat('agent', 'start_epsilon', fallback=1.0)
    CONFIG['end_epsilon'] = _configparser.getfloat('agent', 'end_epsilon', fallback=0.1)
    CONFIG['decay_steps'] = _configparser.getint('agent', 'decay_steps', fallback=300000)

    # training
    CONFIG['steps'] = _configparser.getint('agent', 'steps', fallback=5000)
    CONFIG['train_max_episode_len'] = _configparser.getint('agent', 'train_max_episode_len', fallback=100)
    CONFIG['eval_n_episodes'] = _configparser.getint('agent', 'eval_n_episodes', fallback=10)
    CONFIG['eval_interval'] = _configparser.getint('agent', 'eval_interval', fallback=500)
    CONFIG['use_tensorboard'] = _configparser.getboolean('agent', 'use_tensorboard', fallback=False)

    # eval
    CONFIG['save_eval'] = _configparser.getboolean('agent', 'save_eval', fallback=False)
    CONFIG['pred_bboxes'] = _configparser.get('agent', 'pred_bboxes', fallback='./pred_bboxes.npy')
    CONFIG['pred_labels'] = _configparser.get('agent', 'pred_labels', fallback='./pred_labels.npy')
    CONFIG['pred_scores'] = _configparser.get('agent', 'pred_scores', fallback='./pred_scores.npy')
    CONFIG['gt_bboxes'] = _configparser.get('agent', 'gt_bboxes', fallback='./gt_bboxes.npy')
    CONFIG['gt_labels'] = _configparser.get('agent', 'gt_labels', fallback='./gt_labels.npy')
    CONFIG['iou_threshold'] = _configparser.getfloat('agent', 'iou_threshold', fallback=0.5)

    # if set, override config w/ command line arguments
    for key in CONFIG:
        _argparser.add_argument('--{}'.format(key), type=type(CONFIG[key]))
        args, _ = _argparser.parse_known_args()
        override =  vars(args)[key]
        if override:
            CONFIG[key] = override

    return CONFIG


def write_config(path=None):
    if not path:
        path = CONFIG['resultdir_path']
    cfg = ConfigParser()
    cfg.read_dict({'agent': CONFIG})

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'config.ini'), 'w') as f:
        cfg.write(f)

    print('Saved configuration file to {}'.format(path))


def print_config():
    print('Running w/ config:\n' + json.dumps(CONFIG, indent=4))


load_config()
