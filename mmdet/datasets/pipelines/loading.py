import os.path as osp
import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..utils import prepare_rf, random_scale, zero_pad
from ..registry import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = mmcv.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadColorAndDepthFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        color_path = osp.join(results['img_prefix'],
                              results['color_path'])
        depth_path = osp.join(results['img_prefix'],
                              results['depth_path'])

        color_img = mmcv.imread(color_path)
        depth_img = mmcv.imread(depth_path, -1)
        if len(depth_img.shape) == 3:
            depth_img = np.uint8(depth_img[:, :, 1]*256) + np.uint8(depth_img[:, :, 2])
        depth_img = depth_img.astype(np.uint8)

        if self.to_float32:
            color_img = color_img.astype(np.float32)
            depth_img = depth_img.astype(np.float32)

        assert color_img.shape[:2] == depth_img.shape[:2]
        results['img_path'] = color_path
        results['depth_path'] = color_path
        results['img'] = color_img
        results['depth'] = depth_img
        results['img_shape'] = color_img.shape
        results['ori_shape'] = color_img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotationsNOCS(object):
    def __init__(self,
                 with_mask=True,
                 with_coord=False):
        self.with_mask = with_mask
        self.with_coord = with_coord

    @staticmethod
    def _load_masks(results):
        mask_path = osp.join(results['img_prefix'],
                             results['mask_path'])
        gt_masks = mmcv.imread(mask_path)[:, :, 2]
        results['gt_masks'] = gt_masks

        return results

    @staticmethod
    def _load_coords(results):
        coord_path = osp.join(results['img_prefix'],
                              results['coord_path'])
        gt_coords = mmcv.imread(coord_path)[:, :, :3]
        gt_coords = gt_coords[:, :, (2, 1, 0)]
        results['gt_coords'] = gt_coords

        return results

    @staticmethod
    def process_data(results):
        cdata = results['gt_masks']
        cdata = np.array(cdata, dtype=np.int32)

        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata==255] = -1
        assert(np.unique(cdata).shape[0] < 20)

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = results['gt_coords']
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        meta_path = osp.join(results['img_prefix'],
                             results['meta_path'])
        obj_model_dir = results['obj_model_dir']
        with open(meta_path, 'r') as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')

            if len(words) == 3:
                ## real scanned objs
                if words[2][-3:] == 'npz':
                    npz_path = osp.join(obj_model_dir, words[2])
                    with np.load(npz_path) as npz_file:
                        scale_factor[i, :] = npz_file['scale']
                else:
                    bbox_file = osp.join(obj_model_dir, words[2] + '.txt')
                    scale_factor[i, :] = np.loadtxt(bbox_file)

                scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

            else:
                bbox_file = osp.join(obj_model_dir, words[2], words[3], 'bbox.txt')
                bbox = np.loadtxt(bbox_file)
                scale_factor[i, :] = bbox[0, :] - bbox[1, :]
        i = 0

        # delete ids of background objects and non-existing objects
        inst_dict = results['inst']
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]

        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]

            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            i += 1

        results['gt_masks'] = masks[:, :, :i]
        results['mask_fields'].append('gt_masks')
        coords = coords[:, :, :i, :]
        results['gt_coords'] = np.clip(coords, 0, 1)
        results['coord_fields'].append('gt_coords')

        results['gt_labels'] = class_ids[:i]
        results['scales'] = scales[:i]

        return results

    def __call__(self, results):
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_coord:
            results = self._load_coords(results)

        self.process_data(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_mask={}, with_coord={}, with_meta={})'
                     ).format(self.with_mask, self.with_coord, self.with_meta)
        return repr_str


@PIPELINES.register_module
class ProcessDataNOCS(object):
    def __init__(self):
        pass

    def __call__(self, results):
        cdata = results['gt_masks']
        cdata = np.array(cdata, dtype=np.int32)

        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata==255] = -1
        assert(np.unique(cdata).shape[0] < 20)

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = results['gt_coords']
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        meta_path = osp.join(results['img_prefix'],
                             results['meta_path'])
        obj_model_dir = results['obj_model_dir']
        with open(meta_path, 'r') as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')

            if len(words) == 3:
                ## real scanned objs
                if words[2][-3:] == 'npz':
                    npz_path = osp.join(obj_model_dir, words[2])
                    with np.load(npz_path) as npz_file:
                        scale_factor[i, :] = npz_file['scale']
                else:
                    bbox_file = osp.join(obj_model_dir, words[2] + '.txt')
                    scale_factor[i, :] = np.loadtxt(bbox_file)

                scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

            else:
                bbox_file = osp.join(obj_model_dir, words[2], words[3], 'bbox.txt')
                bbox = np.loadtxt(bbox_file)
                scale_factor[i, :] = bbox[0, :] - bbox[1, :]
        i = 0

        # delete ids of background objects and non-existing objects
        inst_dict = results['inst']
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]

        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]

            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            i += 1

        results['gt_masks'] = [masks[:, :, :i]]
        results['mask_fields'].append('gt_masks')
        coords = coords[:, :, :i, :]
        results['gt_coords'] = [np.clip(coords, 0, 1)]
        results['coord_fields'].append('gt_coords')

        results['gt_labels'] = class_ids[:i]
        results['scales'] = scales[:i]

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([0, 0, 0, 0], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)


@PIPELINES.register_module
class LoadRFImage(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        '''
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None
        '''
        ann = results['ann_info']
        rf_img = ann['rf_img']
        rf_ann = ann['rf_ann']
        cat = ann['cat']

        rf_img = mmcv.imread(osp.join(results['img_prefix'], rf_img))
        rf_img = prepare_rf(rf_img, rf_ann, cat)

        # skip the image if there is no valid gt bbox
        '''
        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          os.path.join(self.img_prefix, img_info['filename']))
            return None
        '''

        # apply transforms
        # flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        # img_scale = random_scale(self.img_scales, self.multiscale_mode)
        # process rf_img, resize to (192, 192)
        #rf_img, _, _, _ = self.img_transform(rf_img, (192, 192))
        #rf_img = zero_pad(rf_img)
        results['rf_img'] = rf_img
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class ExtractBBoxFromMask(object):

    def __init__(self):
        pass

    def __call__(self, results):
        mask = results['gt_masks']
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])

        results['gt_bboxes'] = boxes.astype(np.float32)
        results['bbox_fields'].append('gt_bboxes')

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module
class ExtractCountingPointFromBBox(object):

    def __init__(self):
        pass

    def __call__(self, results):
        bboxes = results['gt_bboxes']
        points = np.zeros([bboxes.shape[-1], 2], dtype=np.int32)
        for i in range(bboxes.shape[-1]):
            bbox = bboxes[i]
            x1, y1, x2, y2 = bbox

            point_x = int((x1 + x2) / 2)
            point_y = int((y1 + y2) / 2)
            points[i] = np.array([point_x, point_y])

        results['gt_points'] = points.astype(np.float32)

        return results

    def __repr__(self):
        return self.__class__.__name__
