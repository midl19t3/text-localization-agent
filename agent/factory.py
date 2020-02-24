import chainer
import chainerrl
from chainer.optimizers import Adam
from chainerrl.replay_buffer import EpisodicReplayBuffer
from chainerrl.explorers import LinearDecayEpsilonGreedy
from chainerrl.q_functions import SingleModelStateQFunctionWithDiscreteAction
from .custom_model import CustomModel


def create_agent(env, config):
    n_actions = env.action_space.n

    # Initialize Q-network for predicting action values
    q_func = SingleModelStateQFunctionWithDiscreteAction(CustomModel(n_actions))
    if config['gpu_id'] != -1:
        q_func = q_func.to_gpu(config['gpu_id'])

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = Adam(
        eps=config['epsilon'],
        amsgrad=True,
        alpha=config['learning_rate'])
    optimizer.setup(q_func)

    # Use epsilon-greedy for exploration
    explorer = LinearDecayEpsilonGreedy(
        start_epsilon=config['start_epsilon'],
        end_epsilon=config['end_epsilon'],
        decay_steps=config['decay_steps'],
        random_action_func=env.action_space.sample)

    # DQN uses Experience Replay. Specify a replay buffer and its capacity.
    replay_buffer = EpisodicReplayBuffer(capacity=config['replay_buffer_capacity'])

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DQN(
        q_func,
        optimizer,
        replay_buffer,
        config['gamma'],
        explorer,
        gpu=config['gpu_id'],
        replay_start_size=config['replay_start_size'],
        update_interval=config['update_interval'],
        target_update_interval=config['target_update_interval'])

    return agent
