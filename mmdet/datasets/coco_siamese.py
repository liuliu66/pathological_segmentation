import os
import numpy as np
import warnings
from pycocotools.coco import COCO
import mmcv
import copy

from mmcv.parallel import DataContainer as DC
from .custom import CustomDataset
from .registry import DATASETS
from .utils import to_tensor, random_scale, zero_pad, \
    prepare_rf, prepare_rf_test


@DATASETS.register_module
class CocoSiameseDataset(CustomDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False
                 ):
        super(CocoSiameseDataset, self).__init__(ann_file,
                                                 pipeline,
                                                 data_root,
                                                 img_prefix,
                                                 seg_prefix,
                                                 proposal_file,
                                                 test_mode)
        self.with_mask = True
        self.with_crowd = True
        self.with_label = True

    def load_annotations(self, ann_file, test_mode=False):
        self.coco = COCO(ann_file)
        catids = self.coco.getCatIds()

        # filter test and tarin categories, 0 for train, 1 for test
        self.cats = []
        for _ in range(2):
            self.cats.append([])
        for i in range(len(catids)):
            if (i + 1) % 4 == 0:
                self.cats[1].append(catids[i])
            else:
                self.cats[0].append(catids[i])

        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()

        # split the dataset with correct categories in different mode
        img_infos = []
        for i in self.img_ids:
            if test_mode:
                info = self.coco.loadImgs([i])[0]
                info['filename'] = info['file_name']
                img_anns_ids = self.coco.getAnnIds(imgIds=i)
                img_anns = self.coco.loadAnns(img_anns_ids)
                img_cats = list()
                for img_ann in img_anns:
                    if img_ann['category_id'] in img_cats:
                        continue
                    elif img_ann['category_id'] in self.cats[1]:
                        img_cats.append(img_ann['category_id'])
                        tmp_info = copy.deepcopy(info)
                        tmp_info['category_id'] = img_ann['category_id']
                        img_infos.append(tmp_info)
                    else:
                        continue
            else:
                info = self.coco.loadImgs([i])[0]
                info['filename'] = info['file_name']
                img_anns_ids = self.coco.getAnnIds(imgIds=i)
                img_anns = self.coco.loadAnns(img_anns_ids)
                cat_flag = False
                for img_ann in img_anns:
                    if img_ann['category_id'] in self.cats[0]:
                        cat_flag = True
                        break
                if cat_flag:
                    img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)

        # different process in different mode
        if not self.test_mode:
            return self._parse_ann_info(ann_info, img_id, self.with_mask)
        else:
            category_id = self.img_infos[idx]['category_id']
            return self._parse_test_ann_info(ann_ids, ann_info, img_id,
                                             category_id, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_test_ann_info(self,
                             img_ann_ids,
                             ann_info,
                             img_id,
                             category_id,
                             with_mask=True):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        cat_ann_ids = self.coco.getAnnIds(catIds=[category_id])
        rf_ann = dict()

        # choose a cate randomly
        while True:
            tmp_id = np.random.randint(0, len(cat_ann_ids))
            if cat_ann_ids[tmp_id] in img_ann_ids:
                continue
            else:
                ann = self.coco.loadAnns([cat_ann_ids[tmp_id]])[0]
                rf_ann = ann
                break

        rf_img_file = self.coco.loadImgs(rf_ann['image_id'])[0]['file_name']
        rf_img = mmcv.imread(os.path.join(self.img_prefix, rf_img_file))
        cls_name = self.CLASSES[self.cat2label[ann['category_id']] - 1]
        rf_img = prepare_rf_test(rf_img, rf_ann, cls_name)

        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            elif ann['category_id'] == category_id:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            else:
                continue
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            rf_img=rf_img,
            img_id=img_id,
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore)
        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def _parse_ann_info(self, ann_info, img_id, with_mask=True):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.

        # choose category in train split
        i = 0
        index = np.random.randint(len(ann_info))
        cat = ann_info[index]['category_id']
        while cat not in self.cats[i]:
            index = np.random.randint(len(ann_info))
            cat = ann_info[index]['category_id']

        rf_ids = self.coco.getImgIds(catIds=[cat])
        rf_id = rf_ids[np.random.randint(0, len(rf_ids))]
        while rf_id == img_id:
            rf_id = rf_ids[np.random.randint(0, len(rf_ids))]

        rf_ann = self.coco.loadAnns(
            self.coco.getAnnIds(imgIds=rf_id, iscrowd=False))
        while len(rf_ann) == 0:
            rf_ann = self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=rf_id, iscrowd=False))

        rf_img_file = self.coco.loadImgs([rf_id])[0]['file_name']
        #rf_img = mmcv.imread(os.path.join(self.img_prefix, rf_img_file))
        #rf_img = prepare_rf(rf_img, rf_ann, cat)

        gt_masks_ann = []
        #gt_masks = []
        #gt_mask_polys = []
        #gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            elif ann['category_id'] == cat:
                gt_bboxes.append(bbox)
                gt_labels.append(1)
                gt_masks_ann.append(ann['segmentation'])
            else:
                continue
            #if with_mask:
                #gt_masks.append(self.coco.annToMask(ann))
                #mask_polys = [
                    #p for p in ann['segmentation'] if len(p) >= 6
                #]  # valid polygons have >= 3 points (6 coordinates)
                #poly_lens = [len(p) for p in mask_polys]
                #gt_mask_polys.append(mask_polys)
                #gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = rf_img_file.replace('jpg', 'png')

        ann = dict(
            rf_img=rf_img_file,
            rf_ann=rf_ann,
            cat=cat,
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        #if with_mask:
            #ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            #ann['mask_polys'] = gt_mask_polys
            #ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                os.path.join(self.seg_prefix,
                             img_info['file_name'].replace('jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            rf_img=rf_img,
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        # Todo process rf_img in GPU
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(os.path.join(self.img_prefix, img_info['filename']))
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None, rf_img=None, cat=None, img_id=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack([_proposal, score
                                       ]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            if rf_img is not None:
                rf_img, _, _, _ = self.img_transform(rf_img, (192, 192))
                rf_img = zero_pad(rf_img)
                _img_meta['rf_img'] = rf_img
            if cat is not None:
                # get category and img_id for test
                _img_meta['label'] = cat[0]
                _img_meta['img_id'] = img_id
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        ann = self.get_ann_info(idx)
        img_id = ann['img_id']
        rf_img = ann['rf_img']
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal, rf_img, ann['labels'], img_id)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
