# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from https://github.com/facebookresearch/detr/blob/main/d2/detr/dataset_mapper.py


import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["DetrDatasetMapper"]


def build_transform_gen(cfg, is_train, aug_flip_crop=True):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train and aug_flip_crop:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train and cfg.AUG.COLOR:
        tfm_gens.append(T.RandomBrightness(0.9, 1.1))
        tfm_gens.append(T.RandomSaturation(0.9, 1.1))
        tfm_gens.append(T.RandomContrast(0.9, 1.1))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def snap_to_grid(box, grid_resolution):
    x_min, y_min, x_max, y_max = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    snapped_x_min = torch.round(x_min / grid_resolution) * grid_resolution
    snapped_y_min = torch.round(y_min / grid_resolution) * grid_resolution
    snapped_x_max = torch.round(x_max / grid_resolution) * grid_resolution
    snapped_y_max = torch.round(y_max / grid_resolution) * grid_resolution
    return torch.stack([snapped_x_min, snapped_y_min, snapped_x_max, snapped_y_max])


def adjust_mesh(bboxes, labels, eps, grid_resolution):
    xyxy_bboxes = bboxes.clone()
    xyxy_bboxes[:, 2] += xyxy_bboxes[:, 0]
    xyxy_bboxes[:, 3] += xyxy_bboxes[:, 1]

    meshes_inds = torch.argwhere(labels == 8).flatten()
    for ind in meshes_inds:
        xyxy_mesh = xyxy_bboxes[ind].clone()

        masks_labels = (labels == 9)
        masks_min = torch.stack([ xyxy_bboxes[:, i] >= xyxy_mesh[i] - eps for i in range(2) ]).prod(dim=0)
        masks_max = torch.stack([ xyxy_bboxes[:, i] <= xyxy_mesh[i] + eps for i in range(2, 4) ]).prod(dim=0)
        cells_included_mask = (masks_labels * masks_max * masks_min > 0).flatten()
        xyxy_cells = xyxy_bboxes[cells_included_mask]
        if len(xyxy_cells) == 0:
            continue

        min_x, min_y = xyxy_cells[:, 0].min().item(), xyxy_cells[:, 1].min().item()
        max_x, max_y = xyxy_cells[:, 2].max().item(), xyxy_cells[:, 3].max().item()
        xyxy_bboxes[ind] = torch.tensor([min_x, min_y, max_x, max_y])

    new_bbox = snap_to_grid(xyxy_bboxes, grid_resolution).T
    return new_bbox


class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

        self.layoutlmv3 = 'layoutlm' in cfg.MODEL.VIT.NAME
        if self.layoutlmv3:
            # We disable the flipping/cropping augmentation in layoutlmv3 to be consistent with pre-training
            # Note that we do not disable resizing augmentation since the text boxes are also resized/normalized.
            aug_flip_crop = False
        else:
            aug_flip_crop = True

        if cfg.INPUT.CROP.ENABLED and is_train and aug_flip_crop:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.adjust_boxes = cfg.ADJUST_BOXES.USE
        self.eps = cfg.ADJUST_BOXES.EPS
        self.grid_resolution = cfg.ADJUST_BOXES.GRID_RESOLUTION
        self.tfm_gens = build_transform_gen(cfg, is_train, aug_flip_crop)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        if self.adjust_boxes:
            annots = dataset_dict['annotations']
            labels = torch.tensor([ann['category_id'] for ann in annots])
            bboxes = torch.tensor([ann['bbox'] for ann in annots])
            new_bboxes = adjust_mesh(bboxes, labels, self.eps, self.grid_resolution)
            for i in range(len(annots)):
                # В dataset_dict тоже меняется, ибо по ссылке
                annots[i]['bbox'] = new_bboxes[i].tolist()

        if "image" in dataset_dict.keys():
            # (C, W, H) -> (W, H, C)
            image = np.transpose(np.array(dataset_dict["image"]), (1, 2, 0))
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        # input: (224, 224, 3), final is everytime different -- ?
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)),
                dtype=torch.float16
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict.keys():
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # dict_keys(['file_name', 'attention_mask', 'graph_mask', 'width', 'height', 'image_id', 'image', 'instances'])
        return dataset_dict
