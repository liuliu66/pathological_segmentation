import os
import math
import numpy as np
import pickle as cPickle
import matplotlib.pyplot as plt

import mmcv

from tools.ancsh_utils import axis_diff_degree, dist_between_3d_lines

thres_r = 0.2


def joint_params_eval(results):
    num_joints = results[0]['joint_params_gt'].shape[0] - 1
    assert num_joints == len(results[0]['label_map']) - 1
    joint_angle_err = [[] for _ in range(num_joints)]
    joint_dist_err = [[] for _ in range(num_joints)]
    for result in results:
        if 'index_per_point' in result:
            joint_cls_pred = result['index_per_point']
            joint_cls_pred = np.argmax(joint_cls_pred, axis=1)
            for j in result['label_map']:
                if j == 0:
                    continue
                joint_idx = np.where(joint_cls_pred == j)[0]
            if 'points_mean' in result.keys():
                result['P'] += result['points_mean']
        else:
            joint_idx = np.arange(result['joint_axis_per_point'].shape[0])
        for j in result['label_map']:
            if j == 0:
                continue
            offset = result['unitvec_per_point'] * (1 - result['heatmap_per_point'].reshape(-1, 1)) * thres_r
            joint_pts = result['P'] + offset
            joint_axis_pred = np.mean(result['joint_axis_per_point'][joint_idx], axis=0)
            joint_pt_pred = np.mean(joint_pts[joint_idx], axis=0)

            joint_axis_gt = result['joint_params_gt'][j, 3:6]
            joint_pt_gt = result['joint_params_gt'][j, 0:3]

            joint_rdiff = axis_diff_degree(joint_axis_gt, joint_axis_pred)
            joint_tdiff = dist_between_3d_lines(joint_pt_gt,
                                                joint_axis_gt,
                                                joint_pt_pred,
                                                joint_axis_pred)
            joint_angle_err[j-1].append(joint_rdiff)
            joint_dist_err[j-1].append(joint_tdiff)

    return joint_angle_err, joint_dist_err
