import sys
import numpy as np
from chainercv.evaluations.eval_detection_voc import eval_detection_voc, calc_detection_voc_prec_rec

from config import CONFIG, print_config


"""
Set arguments w/ config file (--config) or cli
:pred_bboxes :pred_labels :pred_scores :gt_bboxes :gt_labels :iou_threshold
"""
def evaluate_from_files():
    print_config()

    pred_bboxes = np.load(CONFIG['pred_bboxes'])
    pred_labels = np.load(CONFIG['pred_labels'])
    pred_scores = np.load(CONFIG['pred_scores'])
    gt_bboxes = np.load(CONFIG['gt_bboxes'])
    gt_labels = np.load(CONFIG['gt_labels'])

    map = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh=CONFIG['iou_threshold'])['map']
    prec, rec = calc_detection_voc_prec_rec(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
                                            iou_thresh=CONFIG['iou_threshold'])

    print('IoU threshold used: %f' % CONFIG['iou_threshold'])
    print('Mean average precision (map): %f' % map)
    print('Mean precision: %f' % np.mean(prec[1]))
    print('Mean recall: %f' % np.mean(rec[1]))
    return


if __name__ == '__main__':
    evaluate_from_files()
