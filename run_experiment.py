import os
from config import print_config
from train_agent import train_agent
from eval_agent import evaluate_agent

if __name__ == '__main__':
    print_config()
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Train & evaluate agent from config
    experiment_path = train_agent(os.path.join(base_path, 'experiments'))
    evaluate_agent(experiment_path)
