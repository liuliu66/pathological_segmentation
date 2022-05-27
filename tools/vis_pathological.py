import os
import os.path as osp
import cv2
import numpy as np
import random
from pycocotools.coco import COCO
from tqdm import tqdm
import sys

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import mmcv
import pycocotools.mask as maskUtils
from mmdet.apis import init_detector,inference_detector
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val


def show_result(img_name, result, class_names, score_thr=0.3, out_file=None, edge_color=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img_name)[:, :, ::-1]
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    imshow_det_bboxes(
        img_name,
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file,
        edge_color=edge_color)

def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)

def imshow_det_bboxes(img_name,
                      img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      edge_color=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(img.shape[1] / 200, img.shape[0] / 200)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(img)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        color = edge_color[label_text]

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox_int[0], bbox_int[1]),
                          bbox_int[2] - bbox_int[0],
                          bbox_int[3] - bbox_int[1],
                          fill=False, edgecolor=color,
                          linewidth=3.5))
        if len(bbox) > 4:
            label_text = 'Defective Area'
            label_text += '|{:.02f}'.format(bbox[-1])
        ax.text(
            bbox[0], bbox[1] - 2,
            label_text,
            fontsize=20,
            family='serif',
            bbox=dict(
                facecolor=color, alpha=0.5, pad=0, edgecolor='none'),
            color='white')

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        fig.savefig(out_file, dpi=100)
        plt.close('all')


if __name__ == '__main__':
    config_file = 'configs/pathological_segmentation/mask_rcnn_r50_fpn_1x.py'
    checkpoint_file = 'work_dirs/pathological_segmentation/mask_rcnn_r50_fpn_1x/epoch_24.pth'

    model = init_detector(config_file, checkpoint_file)

    colors = sns.hls_palette(len(model.CLASSES), l=.3, s=1)
    edge_color = {}
    for i, cls in enumerate(model.CLASSES):
        edge_color[cls] = colors[i]
    print(model.CLASSES)

    image_dir = 'data/pathological_data/images'
    ann_file = 'data/pathological_data/test.json'
    save_dir = 'demo/pathological_visualizition/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    coco = COCO(ann_file)
    image_list = coco.getImgIds()
    # random.seed(3)
    #image_list = random.sample(image_list, 5)
    imgs = coco.loadImgs(image_list)

    num = 0
    for img in tqdm(imgs):
        num += 1
        image_name = os.path.join(image_dir, img['file_name'])
        # output_name = os.path.join(save_dir, '{}.png'.format(num))
        output_name = os.path.join(save_dir, img['file_name'].replace('.tif', '.png'))
        result = inference_detector(model, image_name)
        show_result(image_name, result, model.CLASSES, score_thr=0.7,
                    out_file=output_name, edge_color=edge_color)


