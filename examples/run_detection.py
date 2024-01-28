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
import os
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_argument_parser, default_setup, launch

from ditod import MyTrainer, add_vit_config
from model.tokenization_graphlayoutlm_fast import GraphLayoutLMTokenizerFast
from data.dataset_processor import get_dataset_dict


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.SBERSLIDES_DATA_DIR = 'datasets/sber-slides'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    tokenizer = GraphLayoutLMTokenizerFast.from_pretrained(
        os.path.split(cfg.MODEL.WEIGHTS)[0],
        tokenizer_file=None,
        use_fast=True,
        add_prefix_space=True,
        revision="main",
    )

    splits = []
    if args.do_train:
        splits.append('train')
    if args.do_eval or args.do_train:
        splits.append('test')

    getter = lambda split: get_dataset_dict(f"datasets/{args.dataset_name}", split, tokenizer, args)
    for split in splits:
        folder_name = args.dataset_name.replace('-', '') + '_' + split
        DatasetCatalog.register(folder_name, getter(split))


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
    parser.add_argument("--dataset_name", default="sber-slides")
    parser.add_argument("--visual_embed", default=True)
    parser.add_argument("--label_all_tokens", default=False)
    parser.add_argument("--imagenet_default_mean_and_std", default=False)
    parser.add_argument("--input_size", default=224)
    parser.add_argument("--train_interpolation", default="bicubic")
    parser.add_argument("--preprocessing_num_workers", default=None)
    parser.add_argument("--overwrite_cache", default=True)
    parser.add_argument("--do_train", default=True)
    parser.add_argument("--do_eval", default=True)
    parser.add_argument("--do_test", default=True)
    args = parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
