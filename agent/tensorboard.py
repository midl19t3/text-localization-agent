import chainerrl
import logging
import re
import numpy as np
from chainer.backends import cuda


class TensorboardGradientPlotter:
    timing = "pre"
    name = "TensorboardGradientPlotter"
    call_for_each_param = False

    def __init__(self, summary_writer, log_interval):
        self.summary_writer = summary_writer
        self.log_interval = log_interval
        self.iteration = 0

    def __call__(self, optimizer):
        self.iteration += 1

        if self.iteration % self.log_interval != 0:
            return

        for param_name, param in optimizer.target.namedparams(False):
            weights, gradients = cuda.to_cpu(param.array), cuda.to_cpu(param.grad)
            if weights is None or gradients is None:
                return

            self.summary_writer.add_histogram(f'localizer{param_name}/weight', weights, self.iteration)
            self.summary_writer.add_histogram(f'localizer{param_name}/gradients', gradients, self.iteration)


class TensorBoardLoggingStepHook(chainerrl.experiments.StepHook):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer
        return

    def __call__(self, env, agent, step):
        step_count = agent.t
        self.summary_writer.add_scalar('average_q', agent.average_q, step_count)
        self.summary_writer.add_scalar('average_loss', agent.average_loss, step_count)

class TensorBoardEvaluationLoggingHandler(logging.Handler):
    def __init__(self, summary_writer, agent, eval_run_count, level=logging.NOTSET):
        logging.Handler.__init__(self, level)
        self.summary_writer = summary_writer
        self.agent = agent
        self.eval_run_count = eval_run_count
        self.episode_rewards = np.empty(eval_run_count)
        self.episode_lengths = np.empty(eval_run_count)
        self.episode_ious = np.empty(eval_run_count)
        self.episode_max_ious = np.empty(eval_run_count)

    def emit(self, record):
        match_new_best = re.search(r'The best score is updated ([^ ]*) -> ([^ ]*)', record.getMessage())
        if match_new_best:
            new_best_score = match_new_best.group(2)
            step_count = self.agent.t
            self.summary_writer.add_scalar('evaluation_new_best_score', new_best_score, step_count)

        match_reward = re.search(r'evaluation episode ([^ ]*) length:([^ ]*) R:([^ ]*) IoU:([^ ]*) Max_IoU:([^ ]*)', record.getMessage())
        if match_reward:
            episode_number = int(match_reward.group(1))
            episode_length = int(match_reward.group(2))
            episode_reward = float(match_reward.group(3))
            episode_iou = float(match_reward.group(4))
            episode_max_iou = float(match_reward.group(5))

            self.episode_lengths[episode_number] = episode_length
            self.episode_rewards[episode_number] = episode_reward
            self.episode_ious[episode_number] = episode_iou
            self.episode_max_ious[episode_number] = episode_max_iou

            if episode_number == self.eval_run_count - 1:
                step_count = self.agent.t
                self.summary_writer.add_scalar('evaluation_length_mean', np.mean(self.episode_lengths), step_count)
                self.summary_writer.add_scalar('evaluation_reward_mean', np.mean(self.episode_rewards), step_count)
                self.summary_writer.add_scalar('evaluation_reward_median', np.median(self.episode_rewards), step_count)
                self.summary_writer.add_scalar('evaluation_reward_variance', np.var(self.episode_rewards), step_count)
                self.summary_writer.add_scalar('evaluation_iou_mean', np.mean(self.episode_ious), step_count)
                self.summary_writer.add_scalar('evaluation_iou_median', np.median(self.episode_ious), step_count)
                self.summary_writer.add_scalar('evaluation_max_iou_mean', np.mean(self.episode_max_ious), step_count)
