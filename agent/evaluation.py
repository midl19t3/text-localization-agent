import os
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC
from .utils import ensure_folder, to_standard_box


def f1(precision, recall):
    sum = precision + recall
    if sum == 0:
        return 0
    return 2 * (precision * recall) / sum


class EvaluationHook(ABC):
    def set_environment(self, env):
        self.env = env

    def start_episode(self, image_idx):
        pass

    def finish_episode(self, image_idx):
        pass

    def before_step(self):
        pass

    def after_step(self, action, obs, reward, done, info):
        pass


class EpisodeRenderer(EvaluationHook):
    def __init__(self, gif_path):
        self.gif_path = gif_path

    def start_episode(self, image_idx):
        self.frames = []

    def after_step(self, action, obs, reward, done, info):
        img = self.env.render(mode='human', return_as_file=True)
        self.frames.append(img)

    def finish_episode(self, image_idx):
        # Save gif
        print('Visualizing %s' % str(image_idx))
        ensure_folder(self.gif_path)
        save_path = os.path.join(self.gif_path, f'{image_idx}.gif')
        self.frames[0].save(save_path,
            format='GIF',
            append_images=self.frames[1:],
            save_all=True,
            duration=100,
            loop=0
        )


class DetectionMetrics(EvaluationHook):
    def __init__(self, eval_path):
        self.eval_path = eval_path
        self.image_idx = None

        # Map from image indices to predicted bounding box
        self.image_pred_bboxes = {}
        self.image_true_bboxes = {}
        # Map from images indices to list of IoU values at each trigger
        self.image_trigger_ious = {}
        self.image_avg_iou = {}
        # Map from image indices to taken actions
        self.image_actions = {}
        self.image_num_actions = {}

    def start_episode(self, image_idx):
        self.image_idx = image_idx
        self.image_actions[self.image_idx] = []
        self.image_num_actions[self.image_idx] = 0

    def after_step(self, action, obs, reward, done, info):
        self.image_actions[self.image_idx].append(action)
        self.image_num_actions[self.image_idx] += 1

    def finish_episode(self, image_idx):
        # Save bounding box predictions
        self.image_pred_bboxes[image_idx] = [to_standard_box(box) for box in self.env.episode_pred_bboxes]
        self.image_true_bboxes[image_idx] = [to_standard_box(box) for box in self.env.episode_true_bboxes]
        # Track intermediary metrics
        self.image_trigger_ious[image_idx] = self.env.episode_trigger_ious
        if len(self.env.episode_trigger_ious) > 0:
            self.image_avg_iou[image_idx] = sum(self.env.episode_trigger_ious) / len(self.env.episode_trigger_ious)
        else:
            self.image_avg_iou[image_idx] = 0
        print(self.image_trigger_ious)
        print(self.image_avg_iou)


def plot_training_summary(experiment_path, display_interval=2000):
    summary_df = pd.read_csv(os.path.join(experiment_path, 'scores.txt'), sep='	')

    if display_interval is not None:
        # Smoothen plots by averaging over intervals
        max_steps = summary_df['steps'].max()
        bins = [x for x in range(display_interval, max_steps, display_interval)]
        summary_df['step_intervals'] = pd.cut(x=summary_df['steps'], bins=bins)
        cut_df = summary_df.groupby(['step_intervals']).mean()
        summary_df = cut_df

    output_path = os.path.join(experiment_path, 'training')
    ensure_folder(output_path)

    def _plot_linechart(x, y, x_label, y_label, title, filename):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()
        fig.savefig(os.path.join(output_path, filename))

    _plot_linechart(
        summary_df['steps'].values, summary_df['average_loss'].values,
        'Steps', 'Average Loss', 'Training Loss', 'average_loss.png'
    )
    _plot_linechart(
        summary_df['steps'].values, summary_df['average_q'].values,
        'Steps', 'Average Q', 'Training Q', 'average_q.png'
    )
    _plot_linechart(
        summary_df['steps'].values, summary_df['mean'].values,
        'Steps', 'Mean Reward', 'Training Reward (Mean)', 'mean_reward.png'
    )
    _plot_linechart(
        summary_df['steps'].values, summary_df['median'].values,
        'Steps', 'Median Reward', 'Training Reward (Median)', 'median_reward.png'
    )
    _plot_linechart(
        summary_df['steps'].values, summary_df['max'].values,
        'Steps', 'Max Reward', 'Training Reward (Max)', 'max_reward.png'
    )
    _plot_linechart(
        summary_df['steps'].values, summary_df['stdev'].values,
        'Steps', 'Stdev Reward', 'Training Reward (Stdev)', 'stdev_reward.png'
    )

    summary_df.to_csv(os.path.join(output_path, 'scores.csv'))