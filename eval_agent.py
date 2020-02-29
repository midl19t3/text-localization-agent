import os
import sys
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from config import CONFIG, print_config
from agent.datasets import load_dataset
from agent.factory import create_agent, create_env
from agent.utils import ensure_folder
from agent.evaluation import f1, EpisodeRenderer, DetectionMetrics, plot_training_summary

# https://github.com/rafaelpadilla/Object-Detection-Metrics
def init_detection_metrics_lib():
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)
    current_path = os.path.dirname(os.path.realpath(__file__))
    lib_path = os.path.join(current_path, 'detection_metrics')
    if not os.path.exists(lib_path):
        raise ModuleNotFoundError('Object Detection Metrics not installed')
    add_path(lib_path)

init_detection_metrics_lib()

try:
    from BoundingBox import BoundingBox
    from BoundingBoxes import BoundingBoxes
    from Evaluator import *
    from utils import *
except ModuleNotFoundError:
    # So that we don't need to install python-opencv
    pass


def run_agent(agent, env, n_samples, hooks=[]):
    for image_idx in range(n_samples):
        print('Image: %s' % str(image_idx))

        obs = env.reset(image_index=image_idx)
        done = False

        for hook in hooks:
            hook.set_environment(env)
            hook.start_episode(image_idx)

        # Environment will terminate based on max steps per image
        while (not done):
            for hook in hooks:
                hook.before_step()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            for hook in hooks:
                hook.after_step(action, obs, reward, done, info)

        for hook in hooks:
            hook.finish_episode(image_idx)


def evaluate_agent(experiment_path, n_samples=100, agent_dir='best',
    visualize_episodes=True
):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    print_config()

    dataset = load_dataset(CONFIG['dataset'], CONFIG['dataset_path'])

    # Always playout full episodes during testing
    CONFIG['playout_episode'] = True
    CONFIG['premasking'] = False
    env = create_env(dataset, CONFIG, mode='test')

    # Load agent from given path
    agent_path = os.path.join(experiment_path, agent_dir)
    agent = create_agent(env, CONFIG, from_path=agent_path)

    # Plot training summary (if it doesn't exist yet)
    if not os.path.exists(os.path.join(experiment_path, 'training')):
        plot_training_summary(experiment_path)

    # Create new evaluation folder
    eval_dirname = 'evaluation'
    eval_path = os.path.join(experiment_path, eval_dirname)
    ensure_folder(eval_path)

    # Use sampling to speed up evaluation if needed
    sample_size = len(dataset)
    if n_samples is not None and n_samples > -1:
        sample_size = min(sample_size, n_samples)

    collector = DetectionMetrics(eval_path)
    hooks = []
    hooks.append(collector)
    if visualize_episodes:
        gif_path = os.path.join(eval_path, 'episodes')
        hooks.append(EpisodeRenderer(gif_path))

    run_agent(agent, env, sample_size, hooks=hooks)

    print("Write bbox files")
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
    _write_bbox_file('predictions', collector.image_pred_bboxes)
    _write_bbox_file('groundtruths', collector.image_true_bboxes)

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

    true_boxes = _generate_lib_bboxes(BBType.GroundTruth, collector.image_true_bboxes)
    # Set default confidence as .01 for now (since agent doesn't score regions)
    pred_boxes = _generate_lib_bboxes(
        BBType.Detected, collector.image_pred_bboxes, confidence=.01
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

    all_actions = list(itertools.chain(*collector.image_actions.values()))
    action_counter = Counter(all_actions)
    n_actions = len(action_counter.keys())

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
        if len(collector.image_avg_iou) > 0:
            metrics['avg_iou'] = sum(list(collector.image_avg_iou.values())) / len(collector.image_avg_iou)
        else:
            metrics['avg_iou'] = 0

        metrics['total_actions'] = sum(list(collector.image_num_actions.values()))
        if len(collector.image_num_actions) > 0:
            metrics['avg_actions'] = sum(list(collector.image_num_actions.values())) / len(collector.image_num_actions)
        else:
            metrics['avg_actions'] = 0
        print(collector.image_num_actions_per_subepisode)
        avg_actions_subepisode = [sum(x) / len(x) if len(x) else 0 for x in collector.image_num_actions_per_subepisode.values()]
        print(avg_actions_subepisode)
        metrics['mean_avg_actions_subepisode'] = sum(avg_actions_subepisode) / len(avg_actions_subepisode)
        print(metrics['mean_avg_actions_subepisode'])

        for action, count in action_counter.items():
            action_name = str(action)
            metrics[f'total_action_{action_name}'] = count

        iou_metrics[iou_threshold] = metrics

    # Save metrics as CSV
    iou_metrics_df = pd.DataFrame.from_dict(iou_metrics, orient='index')
    iou_metrics_df.index.name = 'iou_threshold'
    iou_metrics_df.to_csv(os.path.join(eval_path, 'metrics.csv'))

    print("Generating plots")

    plots_path = os.path.join(eval_path, 'plots')
    ensure_folder(plots_path)

    # Histogram of agent's actions
    fig, ax = plt.subplots()
    ax.hist(all_actions, bins=n_actions, orientation='horizontal', color='#0504aa')
    ax.set(xlabel='Frequency (Total)', ylabel='Action', title='Agent Actions')
    fig.savefig(os.path.join(plots_path, 'action_hist.png'))

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

    images_path = os.path.join(eval_path, 'images')
    ensure_folder(images_path)

    for image_idx in range(sample_size):
        image_path, _ = dataset.get(image_idx, as_image=False)
        image_name = dataset.get_image_name(image_idx)
        image = cv2.imread(image_path)
        image = all_boxes.drawAllBoundingBoxes(image, image_name)
        image_fname = Path(image_path).name
        cv2.imwrite(os.path.join(images_path, image_fname), image)
        print('Image %s created successfully!' % image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Path to experiment folder')
    parser.add_argument('-a', '--agent', type=str, help='Name of agent directory in experiment folder', default='best')
    parser.add_argument('-n', type=int, help='Number of images to be used for evaluation', default=100)
    parser.add_argument('-v', '--visualize', action='store_true', help='Whether episodes should be saved as GIFs', default=True)
    args, _ = parser.parse_known_args()

    evaluate_agent(
        args.experiment, n_samples=args.n, agent_dir=args.agent,
        visualize_episodes=args.visualize
    )
