import os
import numpy as np
from text_localization_environment import TextLocEnv
from chainerrl.links.mlp import MLP
from chainerrl.links import Sequence
from chainerrl.experiments.train_agent import train_agent_with_evaluation
import chainer
import chainerrl
import logging
import sys
from tb_chainer import SummaryWriter
import time
import re
import chainer.computational_graph as c

from custom_model import CustomModel
from config import CONFIG, print_config
from datasets import get_dataset


"""
Set arguments w/ config file (--config) or cli
:imagefile_path :boxfile_path
"""
def main():
    print_config()

    dataset_id = CONFIG['dataset']
    dataset = get_dataset(dataset_id)(CONFIG['dataset_path'])
    image_paths, bounding_boxes = dataset.data()

    env = TextLocEnv(image_paths, bounding_boxes)
    m = CustomModel(10)
    vs = [m(env.reset())]
    g = c.build_computational_graph(vs)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())

if __name__ == '__main__':
    main()
