#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from GraphLayoutLM.ditod import MyTrainer
from GraphLayoutLM.ditod import add_vit_config

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SBERSLIDES_DATA_DIR = 'datasets/sber-slides'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # TODO: load graph dataset from sber-slides/convert to format
    register_coco_instances(
        "sberslides_train", {},
        cfg.SBERSLIDES_DATA_DIR_TRAIN + '/train.json',
        cfg.SBERSLIDES_DATA_DIR
    )

    register_coco_instances(
        "sberslides_val", {},
        cfg.SBERSLIDES_DATA_DIR_TEST + '/val.json',
        cfg.SBERSLIDES_DATA_DIR
    )

    register_coco_instances(
        "sberslides_test", {},
        cfg.SBERSLIDES_DATA_DIR_TEST + '/val.json',
        cfg.SBERSLIDES_DATA_DIR
    )

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
