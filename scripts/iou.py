import os
import numpy as np
from text_localization_environment import TextLocEnv
from chainerrl.links.mlp import MLP
from chainerrl.links import Sequence
from chainerrl.experiments.train_agent import train_agent_with_evaluation
import chainer
import chainerrl
import sys
from collections import defaultdict
from PIL import Image

from custom_model import CustomModel
from config import CONFIG, print_config


ACTION_MEANINGS = {
    0: 'right',
    1: 'left',
    2: 'up,',
    3: 'down',
    4: 'bigger',
    5: 'smaller',
    6: 'fatter',
    7: 'taller',
    8: 'trigger'
}

"""
Set arguments w/ config file (--config) or cli
:gpu_id :imagefile_path :boxfile_path :resultdir_path :start_epsilon :end_epsilon :decay_steps \
:replay_buffer_capacity :gamma :replay_start_size :update_interval :target_update_interval :steps \
:steps :eval_n_episodes :train_max_episode_len :eval_interval
"""
def main():
    print_config()

    relative_paths = np.loadtxt(CONFIG['imagefile_path'], dtype=str)
    images_base_path = os.path.dirname(CONFIG['imagefile_path'])
    absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
    bboxes = np.load(CONFIG['boxfile_path'], allow_pickle=True)

    env = TextLocEnv(absolute_paths, bboxes, -1)
    q_func = chainerrl.q_functions.SingleModelStateQFunctionWithDiscreteAction(CustomModel(9))
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=CONFIG['replay_buffer_capacity'])

    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0,
        random_action_func=env.action_space.sample)

    agent = chainerrl.agents.DQN(
        q_func,
        optimizer,
        replay_buffer,
        CONFIG['gamma'],
        explorer,
        gpu=CONFIG['gpu_id'],
        replay_start_size=CONFIG['replay_start_size'],
        update_interval=CONFIG['update_interval'],
        target_update_interval=CONFIG['target_update_interval'])

    agent.load(CONFIG['resultdir_path'] + '/best')
    actions = defaultdict(int)
    with open('iou.txt', 'w') as f:
        i = 0
        for j in range(100):
            obs = env.reset()
            done = False
            while (not done) and i < 100:
                #print(i,j)
                action = agent.act(obs)
                actions[ACTION_MEANINGS[action]] += 1
                obs, reward, done, info = env.step(action)

                print(ACTION_MEANINGS[action], reward, done, info)
                if done:
                    f.write(f'{env.iou}\n')
                #input()
                i += 1


if __name__ == '__main__':
    main()
