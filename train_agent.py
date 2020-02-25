import os
import sys
import datetime
import logging
from chainerrl.misc.random_seed import set_random_seed
from chainerrl.experiments.train_agent import train_agent_with_evaluation
from tb_chainer import SummaryWriter
from config import CONFIG, write_config, print_config
from agent.datasets import load_dataset
from agent.factory import create_agent
from agent.tensorboard import TensorBoardLoggingStepHook, TensorBoardEvaluationLoggingHandler
from agent.utils import ensure_folder
from agent.evaluation import plot_training_summary
from text_localization_environment import TextLocEnv


def train_agent(experiments_dir='./experiments'):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    print_config()

    dataset = load_dataset(CONFIG['dataset'], CONFIG['dataset_path'])
    env = TextLocEnv(dataset.image_paths, dataset.bounding_boxes)
    agent = create_agent(env, CONFIG)

    # Seeding for reproducable experiments
    set_random_seed(CONFIG['seed_agent'], gpus=[CONFIG['gpu_id']])
    env.seed(CONFIG['seed_environment'])

    # Prepare experiment directory
    now_date = datetime.datetime.now()
    timestr = now_date.strftime("%Y-%m-%d+%H-%M-%S")
    experiment_path = os.path.join(experiments_dir, CONFIG['experiment_id'] + "_" + timestr)
    ensure_folder(experiment_path)
    write_config(experiment_path)

    step_hooks = []
    logger = None

    if CONFIG['use_tensorboard']:
        tensorboard_path = os.path.join(experiment_path, "tensorboard")
        ensure_folder(tensorboard_path)
        eval_run_count = 10
        writer = SummaryWriter(tensorboard_path)
        step_hooks = [TensorBoardLoggingStepHook(writer)]
        handler = TensorBoardEvaluationLoggingHandler(writer, agent, eval_run_count)
        logger = logging.getLogger()
        logger.addHandler(handler)

    train_agent_with_evaluation(
        agent,
        env,
        steps=CONFIG['steps'],  # Train the agent for no of steps
        eval_n_episodes=CONFIG['eval_n_episodes'],  # Episodes are sampled for each evaluation
        eval_n_steps=None,
        train_max_episode_len=CONFIG['train_max_episode_len'],  # Maximum length of each episodes
        eval_interval=CONFIG['eval_interval'],  # Evaluate the agent after every no of steps
        outdir=experiment_path,  # Save everything to experiment directory
        step_hooks=step_hooks,
        logger=logger)

    # Save the final model
    agent_classname = agent.__class__.__name__[:10]
    agent_path = os.path.join(experiment_path, "agent" + "_" + agent_classname)
    ensure_folder(agent_path)
    agent.save(agent_path)

    # Plot training summary
    if not os.path.exists(os.path.join(experiment_path, 'training')):
        plot_training_summary(experiment_path)

    return experiment_path

if __name__ == '__main__':
    _ = train_agent()
