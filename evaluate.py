import os
import sys
import datetime
import numpy as np
import pandas as pd
import json
import chainer
import chainerrl
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from text_localization_environment import TextLocEnv
from PIL import Image

from custom_model import CustomModel
from config import CONFIG, print_config
from actions import ACTIONS

def init_detection_metrics_lib():
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)
    current_path = os.path.dirname(os.path.realpath(__file__))
    lib_path = os.path.join(current_path, 'detection_metrics')
    add_path(lib_path)

init_detection_metrics_lib()

try:
    from BoundingBox import BoundingBox
    from BoundingBoxes import BoundingBoxes
    from Evaluator import *
    from utils import *
except ModuleNotFoundError:
    print('Object Detection Library not initialized correctly')


class Dataset:
    def __init__(self, image_paths, true_bboxes):
        self.image_paths = image_paths
        self.true_bboxes = true_bboxes

    @classmethod
    def from_json(cls, jsonfile_path):
        absolute_paths, bboxes = [], []
        images_base_path = os.path.dirname(jsonfile_path)
        with open(jsonfile_path) as f:
            data = json.load(f)
            for example in data:
                absolute_path = os.path.join(images_base_path, example['file_name'])
                if os.path.exists(absolute_path):
                    absolute_paths.append(absolute_path)
                    bboxes.append(list(example['bounding_boxes']))
        return cls(absolute_paths, bboxes)

    @classmethod
    def from_numpy(cls, imagefile_path, boxfile_path):
        images_base_path = os.path.dirname(imagefile_path)
        relative_paths = np.loadtxt(imagefile_path, dtype=str)
        absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
        if type(absolute_paths) is not list:
            absolute_paths = [image_paths]
        bboxes = np.load(boxfile_path, allow_pickle=True)
        return cls(absolute_paths, bboxes)

    def get(self, idx, as_image=True):
        image = self.image_paths[idx]
        if as_image:
            image = Image.open(image)
        true_bboxes = self.true_bboxes[idx]
        return image, true_bboxes

    def get_image_name(self, idx):
        image_path, _ = self.get(idx, as_image=False)
        image_fname = Path(image_path).name
        image_name = image_fname.split('.')[0]
        return image_name

    def random_sample(self, as_image=True):
        random_index = np.random.randint(len(self.image_paths))
        return self.get(random_idx, as_image=as_image)

    def __len__(self):
        return len(self.image_paths)


def create_agent_with_environment(actions, dataset, agentdir_path,
    env_mode='test', gpu_id=-1, max_steps_per_image=200,
    # Training params needed to initialize chainer, but shouldn't matter for testing
    replay_buffer_capacity=20000, gamma=0.95, replay_start_size=100,
    update_interval=1, target_update_interval=100
):
    num_actions = len(actions)
    env = TextLocEnv(
        dataset.image_paths, dataset.true_bboxes, -1, mode=env_mode,
        premasking=False, playout_episode=True,
        max_steps_per_image=max_steps_per_image
    )
    q_func = chainerrl.q_functions.SingleModelStateQFunctionWithDiscreteAction(
        CustomModel(num_actions))
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(replay_buffer_capacity)

    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0,
        random_action_func=env.action_space.sample)

    agent = chainerrl.agents.DQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=gpu_id,
        replay_start_size=replay_start_size,
        update_interval=update_interval,
        target_update_interval=target_update_interval)
    agent.load(agentdir_path)

    return agent, env


def ensure_folder(dir_path, exist_ok=True):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=exist_ok)

def f1(precision, recall):
    sum = precision + recall
    if sum == 0:
        return 0
    return 2 * (precision * recall) / sum

"""
Set arguments w/ config file (--config) or cli
:gpu_id :imagefile_path :boxfile_path :resultdir_path :start_epsilon :end_epsilon :decay_steps \
:replay_buffer_capacity :gamma :replay_start_size :update_interval :target_update_interval :steps \
:steps :eval_n_episodes :train_max_episode_len :eval_interval
"""
def main(eval_dirname='evaluations', viz_dirname='episodes', images_dirname='images', plots_dirname='plots', max_sample_size=200):
    print_config()

    if CONFIG['jsonfile_path']:
        dataset = Dataset.from_json(CONFIG['jsonfile_path'])
    else:
        dataset = Dataset.from_numpy(CONFIG['imagefile_path'], CONFIG['boxfile_path'])

    agent, env = create_agent_with_environment(
        ACTIONS, dataset, CONFIG['agentdir_path']
    )

    # Create new evaluation folder
    now_date = datetime.datetime.now()
    now_dirname = now_date.strftime("%m-%d-%Y+%H-%M-%S")
    eval_path = os.path.join(eval_dirname, now_dirname)
    ensure_folder(eval_path)

    print("Running localizations")
    # Map from image indices to predicted bounding box
    image_pred_bboxes = {}
    image_true_bboxes = {}
    # Map from images indices to list of IoU values at each trigger
    image_trigger_ious = {}
    image_avg_iou = {}

    # Use sampling to speed up evaluation if needed
    sample_size = len(dataset)
    if max_sample_size is not None and max_sample_size > -1:
        sample_size = min(sample_size, max_sample_size)

    for image_idx in range(sample_size):
        obs = env.reset(image_index=image_idx)
        done = False
        # Environment will terminate based on max steps per image
        while (not done):
            print(env.current_step)
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        # Save bounding box predictions
        image_pred_bboxes[image_idx] = [TextLocEnv.to_standard_box(box) for box in env.episode_pred_bboxes]
        image_true_bboxes[image_idx] = [TextLocEnv.to_standard_box(box) for box in env.episode_true_bboxes]
        print(image_pred_bboxes[image_idx])
        print(image_true_bboxes[image_idx])
        # Track intermediary metrics
        image_trigger_ious[image_idx] = env.episode_trigger_ious
        if len(env.episode_trigger_ious) > 0:
            image_avg_iou[image_idx] = sum(env.episode_trigger_ious) / len(env.episode_trigger_ious)
        else:
            image_avg_iou[image_idx] = 0
        print(image_trigger_ious)
        print(image_avg_iou)

    print("Write bbox files for Object Detection Metrics")

    def _write_bbox_file(name, bbox_map):
        dir_path = os.path.join(eval_path, name)
        ensure_folder(dir_path)
        for image_idx, bboxes in bbox_map.items():
            image_name = dataset.get_image_name(image_idx)
            print(image_name)
            image_txt = ''
            for bbox in bboxes:
                image_txt += 'text ' # object class name
                # Ensure bounding boxes are saved as integers
                image_txt += str(int(bbox[0])) + ' '
                image_txt += str(int(bbox[1])) + ' '
                image_txt += str(int(bbox[2])) + ' '
                image_txt += str(int(bbox[3])) + ' '
                image_txt += '\n'
            print(image_txt)
            txt_fpath = os.path.join(dir_path, f'{image_name}.txt')
            with open(txt_fpath, 'w+') as f:
                f.write(image_txt)

    _write_bbox_file('detections', image_pred_bboxes)
    _write_bbox_file('groundtruths', image_true_bboxes)

    print("Evaluating predictions against ground truth")

    def _generate_lib_bboxes(bb_type, bbox_map, confidence=None):
        boxes = []
        for image_idx, bboxes in bbox_map.items():
            image_name = dataset.get_image_name(image_idx)
            for bbox in bboxes:
                box = BoundingBox(
                    image_name,
                    'text',  # object class name
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    typeCoordinates=CoordinatesType.Absolute,
                    classConfidence=confidence,
                    bbType=bb_type,
                    format=BBFormat.XYX2Y2
                )
                boxes.append(box)
        return boxes

    true_boxes = _generate_lib_bboxes(BBType.GroundTruth, image_true_bboxes)
    # Set default confidence as .01 for now (since agent doesn't score regions)
    pred_boxes = _generate_lib_bboxes(
        BBType.Detected, image_pred_bboxes, confidence=.01
    )

    all_boxes = BoundingBoxes()
    for bbox in pred_boxes:
        all_boxes.addBoundingBox(bbox)
    for bbox in true_boxes:
        all_boxes.addBoundingBox(bbox)

    evaluator = Evaluator()
    # Mapping from IoU treshold to metrics calculated at this threshold
    iou_metrics = {}
    iou_thresholds = [round(x, 2) for x in np.arange(0, 1, .05)]

    for iou_threshold in iou_thresholds:
        metrics_per_class = evaluator.GetPascalVOCMetrics(
            all_boxes,
            IOUThreshold=iou_threshold,
            method=MethodAveragePrecision.EveryPointInterpolation
        )
        text_metrics = metrics_per_class[0]  # class = 'text'
        metrics = {
            'precision': text_metrics['precision'][-1],
            'recall': text_metrics['recall'][-1],
            'ap': text_metrics['AP'],
            'num_p_total': text_metrics['total positives'],
            'num_tp': text_metrics['total TP'],
            'num_fp': text_metrics['total FP'],
        }
        metrics['f1'] = f1(metrics['precision'], metrics['recall'])
        metrics['avg_iou'] = sum(list(image_avg_iou.values())) / len(image_avg_iou)
        iou_metrics[iou_threshold] = metrics

    # Save metrics as CSV
    iou_metrics_df = pd.DataFrame.from_dict(iou_metrics, orient='index')
    iou_metrics_df.index.name = 'iou_threshold'
    iou_metrics_df.to_csv(os.path.join(eval_path, 'metrics.csv'))

    print("Plotting metrics")

    plots_path = os.path.join(eval_path, plots_dirname)
    ensure_folder(plots_path)

    # Precision-Recall curves at different IoU thresholds
    for iou_threshold in [0.5, 0.75]:
        iou_fname_str = str(iou_threshold).replace('.', '')
        plot_path = os.path.join(plots_path, f'ap_{iou_fname_str}')
        ensure_folder(plot_path)
        evaluator.PlotPrecisionRecallCurve(
            all_boxes,
            IOUThreshold=iou_threshold,
            method=MethodAveragePrecision.EveryPointInterpolation,
            showAP=True,
            showInterpolatedPrecision=True,
            savePath=plot_path,
            showGraphic=False
        )

    # Recall-IoU curve
    x = iou_metrics_df.index.values
    y = iou_metrics_df['recall'].values
    fig, ax = plt.subplots()
    ax.plot(x, y, '-o')
    ax.set(xlabel='Intersection over Union (IoU)', ylabel='Recall', title='Recall-IoU')
    ax.grid()
    fig.savefig(os.path.join(plots_path, 'recall_iou.png'))

    # Precision-IoU curve
    x = iou_metrics_df.index.values
    y = iou_metrics_df['precision'].values
    fig, ax = plt.subplots()
    ax.plot(x, y, '-o')
    ax.set(xlabel='Intersection over Union (IoU)', ylabel='Precision', title='Precision-IoU')
    ax.grid()
    fig.savefig(os.path.join(plots_path, 'precision_iou.png'))

    print("Drawing images with predictions and ground truths")

    images_path = os.path.join(eval_path, images_dirname)
    ensure_folder(images_path)

    for image_idx in range(sample_size):
        image_path, _ = dataset.get(image_idx, as_image=False)
        image_name = dataset.get_image_name(image_idx)
        image = cv2.imread(image_path)
        image = all_boxes.drawAllBoundingBoxes(image, image_name)
        image_fname = Path(image_path).name
        cv2.imwrite(os.path.join(images_path, image_fname), image)
        print('Image %s created successfully!' % image_name)

    # Visualize agent behaviour on example images
    print("Visualizing episodes")

    viz_path = os.path.join(eval_path, viz_dirname)
    ensure_folder(viz_path)

    for image_idx in range(sample_size):
        obs = env.reset(image_index=image_idx)
        done = False
        frames = []
        # Environment will terminate based on max steps per image
        while (not done):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            img = env.render(mode='human', return_as_file=True)
            frames.append(img)
        gif_path = os.path.join(viz_path, f'{image_idx}.gif')
        frames[0].save(gif_path,
           format='GIF',
           append_images=frames[1:],
           save_all=True,
           duration=100,
           loop=0
       )
       image_name = dataset.get_image_name(image_idx)
       print('Image %s created successfully!' % image_name)

if __name__ == '__main__':
    main()
