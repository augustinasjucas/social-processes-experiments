#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: train_dataset.py
# Created Date: Sunday, July 5th 2020, 1:28:08 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import argparse
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger
import torch
import numpy as np
import constants.paths as paths
from common.initialization import init_torch
from common.model import summarize
from common.utils import configure_logging
from data.datasets import SocialDataset
from data.types import FeatureSet
from lightning.data import SPSocialDataModule
from lightning.processes import SPSystemSocial
from run.utils import (
    add_commmon_args, init_model, init_data, init_ckpt_callbacks, override_hidden_dims

)

def visualize_batch(hparams, batch, index = 0):

    from xrprimer.data_structure.keypoints import Keypoints
    from xrprimer.visualization.keypoints.visualize_keypoints3d import (
        visualize_keypoints3d,
    )

    context = batch.context
    target = batch.target
    
    # leave only one observed context sequence

    keypoints = context.observed[:, 0, :, :]
    
    # remove speaking dimension
    keypoints = keypoints[:, :, :-1]

    keypoints = torch.reshape(keypoints, (keypoints.shape[0], keypoints.shape[1], keypoints.shape[2] // 3, 3))

    keypoints = np.array(keypoints)

    # upsample the keypoints to match time scale (they were sampled every hparams.time_stride frames)
    # keypoints = np.repeat(keypoints, hparams.time_stride, axis=0)

    kp = Keypoints(
        kps=keypoints,
        convention="panoptic",
    )
    visualize_keypoints3d(kp, output_path="tmp"+ str(index) + ".mp4", disable_tqdm=False)




def main() -> None:
    """ Run the main experiment """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip_monitoring", default=False, action="store_true",
                        help="Skip the monitored checkpoint callback")
    parser.add_argument("--override_hid_dims", default=False, action="store_true",
                        help="Override representation dimensions to `hid_dim`")
    parser.add_argument("--hid_dim", type=int, default=1024,
                        help="dimension to override representations")
    parser.add_argument("--out_dir", type=str, help="root output directory")

    parser.add_argument("--paramset_id", type=str, 
                        help="ID of the parameter set, used purely for logging")

    parser = add_commmon_args(parser)
    # Add Trainer args
    parser = Trainer.add_argparse_args(parser)
    # Add model specific args
    parser = SPSystemSocial.add_model_specific_args(parser)
    parser = SPSocialDataModule.add_data_args(parser)
    parser = SocialDataset.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # Override representations dimensions if needed
    if args.override_hid_dims:
        args = override_hidden_dims(args, args.hid_dim)
    
    # In case we use all data, we do not do any custom pooling
    if args.feature_set == FeatureSet.FULL:
        args.no_pool = True

    # Create the output root and update log_file path
    outroot = Path(args.out_dir)
    outroot.mkdir(parents=True, exist_ok=True)
    args.log_file = str(outroot / args.log_file)

    # Setup logging
    configure_logging(args.log_level, args.log_file)

    # Initialize pytorch
    init_torch(args.seed)
    seed_everything(args.seed)

    # Prepare the datasets
    dataset_dir = (Path(__file__).resolve().parent.parent / "artefacts"
                   / "datasets" / args.dataset_root)
    dm = init_data(args, dataset_dir)
    dm.setup("fit")

    # print(" === inspecting dataloader")
    # train_loader = dm.train_dataloader()
    # val_loader = dm.val_dataloader()

    # print("loader created")
    # for i, batch in enumerate(train_loader):


        
    #     print("Batch context observed shape", batch.context.observed.shape)
    #     print("Batch context future ", batch.context.future.shape)
    #     print("Batch target observed shape", batch.target.observed.shape)
    #     print("Batch target future ", batch.target.future.shape)
    #     print("===")
    #     if i > 10:
    #         break
    # print("== done")
    # visualize_batch(args, batch)
    # Initialize the lightning module
    model = init_model(args)

    # Summarize the module and the parameters
    summarize(model)

    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is not None:
        resume_ckpt = str(ckpt_dir / resume_ckpt)

    # Create experiment
    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Create trainer and fit
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=resume_ckpt
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
