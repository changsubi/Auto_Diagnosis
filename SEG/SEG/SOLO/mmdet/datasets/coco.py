import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoDataset(CustomDataset):

    CLASSES = ('L_AP1', 'L_AP2', 'L_AP3', 'L_AP4', 'L_AP5', 'L_AP6', 'L_AP7', 'L_AP8', 'R_AP1', 'R_AP2', 'R_AP3', 'R_AP4', 'R_AP5', 'R_AP6', 'R_AP7', 'R_AP8', 'L_LAT1', 'L_LAT2', 'L_LAT3', 'L_LAT4', 'L_LAT5', 'L_LAT6', 'R_LAT1', 'R_LAT2', 'R_LAT3', 'R_LAT4', 'R_LAT5', 'R_LAT6', 'L_AWB_AP1', 'L_AWB_AP2', 'L_AWB_AP3', 'R_AWB_AP1', 'R_AWB_AP2', 'R_AWB_AP3', 'R_AWB_LAT1', 'R_AWB_LAT2', 'L_AWB_LAT1', 'L_AWB_LAT2', 'L_KWB_AP1', 'L_KWB_AP2', 'L_KWB_AP3', 'R_KWB_AP1', 'R_KWB_AP2', 'R_KWB_AP3', 'R_KWB_LAT1', 'R_KWB_LAT2', 'L_KWB_LAT1', 'L_KWB_LAT2', 'L_T1', 'L_T2', 'L_T3', 'L_T4', 'R_T1', 'R_T2', 'R_T3', 'R_T4', 'L_HAV1', 'L_HAV2', 'R_HAV1', 'R_HAV2')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

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

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
