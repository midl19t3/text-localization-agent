import os
import sys
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
        # Map from image indices to predicted bounding box
        self.image_pred_bboxes = {}
        self.image_true_bboxes = {}
        # Map from images indices to list of IoU values at each trigger
        self.image_trigger_ious = {}
        self.image_avg_iou = {}

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
