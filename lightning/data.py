#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: data.py
# Created Date: Thursday, August 20th 2020, 1:09:56 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import logging
import pickle
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Sequence, Tuple, Type

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from common.utils import EnumAction
from data.datasets import (
    BUCKET_ATTR_NAME, SocialDatasetInterface, SocialDataset, SocialUnpairedContextDataset,
    SocialPairedContextDataset
)
from data.loader import (
    COLLATE_TYPE, BucketSampler, collate_random_context, collate_seq2seq,
    collate_sampled_context, collate_unpaired_context, collate_paired_context
)
from data.preprocessing import BASIC_FEATURES_COLUMNS
from data.types import BucketType, ContextRegime, FeatureSet, ModelType


# Map from feature set to the column names expected in the dataframe
FULL_COLUMNS = [
    'Neck_x', 'Neck_y', 'Neck_z', 
    'Nose_x', 'Nose_y', 'Nose_z', 
    'BodyCenter_x', 'BodyCenter_y', 'BodyCenter_z', 
    'lShoulder_x', 'lShoulder_y', 'lShoulder_z', 
    'lElbow_x', 'lElbow_y', 'lElbow_z', 
    'lWrist_x', 'lWrist_y', 'lWrist_z',
    'lHip_x', 'lHip_y', 'lHip_z',
    'lKnee_x', 'lKnee_y', 'lKnee_z', 
    'lAnkle_x', 'lAnkle_y', 'lAnkle_z',
    'rShoulder_x', 'rShoulder_y', 'rShoulder_z',
    'rElbow_x', 'rElbow_y', 'rElbow_z', 
    'rWrist_x', 'rWrist_y', 'rWrist_z',
    'rHip_x', 'rHip_y', 'rHip_z', 
    'rKnee_x', 'rKnee_y', 'rKnee_z',
    'rAnkle_x', 'rAnkle_y', 'rAnkle_z', 
    'lEye_x', 'lEye_y', 'lEye_z',
    'lEar_x', 'lEar_y', 'lEar_z', 
    'rEye_x', 'rEye_y', 'rEye_z', 
    'rEar_x', 'rEar_y', 'rEar_z', 
    'speaking'
]
FEATURE_FIELDS_MAP = {FeatureSet.HBPS: BASIC_FEATURES_COLUMNS,
                      FeatureSet.HBP: BASIC_FEATURES_COLUMNS[:-1],
                      FeatureSet.FULL: FULL_COLUMNS}


# Column names to standardize
COLUMNS_TO_STDIZE = []

# Percentage of frames of an interaction to use as context
CONTEXT_FRAMES_FRACTION = 0.2


def contextualize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split the dataframe groups into context and target dataframes """
    # Split the groups into context window and target window from which obs,
    # future pairs are computed.
    gdfs = [d for _, d in df.groupby("group_id")]
    ctx_frames = [d.iloc[0].frame
                  + int((d.iloc[-1].frame - d.iloc[0].frame)
                        * CONTEXT_FRAMES_FRACTION)
                  for d in gdfs]
    ctx_df = pd.concat([d.loc[d.frame <= fr] for d, fr in zip(gdfs, ctx_frames)])
    trg_df = pd.concat([d.loc[d.frame > fr] for d, fr in zip(gdfs, ctx_frames)])
    return trg_df, ctx_df


class SPSocialDataModule(pl.LightningDataModule):

    """ Setup the datasets and loaders for the Process Lightning Module """

    def __init__(self, dataset_dir: Path, hparams: Namespace, test_batches: Optional[Sequence] = None) -> None:
        """ Initialize the data module

        Args:
            dataset_dir     --  The root dataset directory
            hparams         --  The arguments for preparing the data
            feature_set     --  The feature set in the data
            test_batches    --  The fixed batch indices for the test
                                sampler to control for random sampling
                                across different model comparisons

        """
        super().__init__()
        self.data_dir = dataset_dir
        self.hparams = hparams
        self.feature_set = hparams.feature_set
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.collate_fn = None
        self.test_batches = test_batches

    def extract_train_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Extract and standardize the train and val dataframes

        Args:
            df  --  The main dataset dataframe

        """
        if not self.hparams.train_all:
            logging.info(
                "[*] Using training fold - {}".format(self.hparams.fold_file)
            )
            with open(self.data_dir/self.hparams.fold_file, "rb") as f:
                fold = pickle.load(f)
        else:
            logging.info("[*] Using entire training set")
            with open(self.data_dir/"train.txt", "r") as f:
                train_groups = [l.strip("\n") for l in f]
            fold = {"train": train_groups, "val": train_groups}
            print("val and train are the same groups!")

        # Extract the train dataframe and standardize
        train_df = df.loc[df.group_id.isin(fold["train"])]
        # Standardize location columns
        std_train_df = train_df.copy()
        std_train_df.loc[:, COLUMNS_TO_STDIZE] = std_train_df.loc[:, COLUMNS_TO_STDIZE].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )
        # Extract the val dataframe and standardize
        val_df = df.loc[df.group_id.isin(fold["val"])].copy()
        val_df.loc[:, COLUMNS_TO_STDIZE] = val_df.loc[:, COLUMNS_TO_STDIZE].apply(
            lambda x: (x - train_df[x.name].mean()) / train_df[x.name].std(),
            axis=0
        )

        # Replace NaN values with 0; useful if datasets have 2D locations
        # and the Z component is a constant
        std_train_df.fillna(0, inplace=True)
        val_df.fillna(0, inplace=True)

        return std_train_df, val_df

    def extract_test_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Load the test dataframe and standardize by train summary stats

        Args:
            df  --  The main dataset dataframe

        """
        # Load train summary statistics to standardize test features
        tr_desc = pd.read_hdf(self.data_dir/"train_description.h5")

        # Load the test group keys
        logging.info("[*] Using entire test set")
        with open(self.data_dir/"test.txt", "r") as f:
            test_groups = [l.strip("\n") for l in f]
        # Extract the train dataframe and standardize
        test_df = df.loc[df.group_id.isin(test_groups)].copy()

        # Standardize location columns
        test_df.loc[:, COLUMNS_TO_STDIZE] = test_df.loc[:, COLUMNS_TO_STDIZE].apply(
            lambda x: (x - tr_desc.loc["mean", x.name]) / tr_desc.loc["std", x.name],
            axis=0
        )
        # Replace NaN values with 0; useful if datasets have 2D locations
        # and the Z component is a constant
        test_df.fillna(0, inplace=True)
        return test_df

    def _init_fixed_ctx_set(
            self, df: pd.DataFrame, dataset_cls: Type, feature_fields: Sequence
        ) -> SocialDatasetInterface:
        """ Initialize a fixed context dataset """
        trg, ctx = contextualize_df(df)
        return dataset_cls(trg, ctx, self.hparams, feature_fields)

    def _init_random_ctx_set(
            self, df: pd.DataFrame, feature_fields: Sequence
        ) -> SocialDatasetInterface:
        """ Initialize a random context dataset """
        return SocialDataset(df, self.hparams, feature_fields)

    def init_datasets(self, stage: Optional[str] = None) -> None:
        """ Initialize the train, val, and test sets """
        # Set the feature columns to extract from the dataframes
        feature_fields = FEATURE_FIELDS_MAP[self.feature_set]

        if self.hparams.context_regime == ContextRegime.FIXED:
            # Choose the dataset type and collate function for the desired model
            if self.hparams.model == ModelType.SOCIAL_PROCESS:
                dataset_cls = SocialUnpairedContextDataset
                self.collate_fn = collate_unpaired_context
            elif self.hparams.model in (ModelType.NEURAL_PROCESS, ModelType.VAE_SEQ2SEQ):
                dataset_cls = SocialPairedContextDataset
                self.collate_fn = collate_paired_context
            dataset_init = lambda x: self._init_fixed_ctx_set(
                x, dataset_cls, feature_fields
            )
        elif self.hparams.context_regime == ContextRegime.RANDOM:
            dataset_init = lambda x: self._init_random_ctx_set(x, feature_fields)
            self.collate_fn = collate_random_context

        # Load the dataset
        df = pd.read_hdf(self.data_dir/self.hparams.data_file)

        if stage == "fit" or stage is None:
            train_df, val_df = self.extract_train_groups(df)
            self.train_set = dataset_init(train_df)
            self.val_set = dataset_init(val_df)

        if stage == "test" or stage is None:
            test_df = self.extract_test_groups(df)
            self.test_set = dataset_init(test_df)

    def compute_samples(self, stage: Optional[str] = None) -> None:
        """ Compute observed, future pairs for the datasets """
        if stage == "fit" or stage is None:
            self.train_set.compute_samples(
                fix_future_len=self.hparams.fix_future_len
            )
            self.val_set.compute_samples(fix_future_len=True)

        if stage == "test" or stage is None:
            self.test_set.compute_samples(fix_future_len=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """ Initialize the datasets and compute observed, future pairs """
        self.init_datasets(stage)
        self.compute_samples(stage)

    def _dataloader(
            self, dataset: SocialDatasetInterface, bucket_type: BucketType,
            collate_fn: COLLATE_TYPE, fixed_batches: Optional[Sequence] = None
        ) -> DataLoader:
        """ Stage agnostic data loader """
        sampler = BucketSampler(dataset, BUCKET_ATTR_NAME[bucket_type],
                                self.hparams.batch_size, fixed_batches=fixed_batches)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn
            # num_workers=self.hparams.ndata_workers
        )

    def train_dataloader(self):
        """ Construct the training data loader """
        loader = None
        if self.train_set is not None:
            if self.hparams.model != ModelType.VAE_SEQ2SEQ:
                collate_fn = self.collate_fn
            else:
                # For meta-learning models, context is sampled as a subset of targets (batch).
                # For the VAE_SEQ2SEQ models, no subsampling is required.
                collate_fn = collate_seq2seq
            loader = self._dataloader(
                self.train_set, self.hparams.bucket_type, collate_fn
            )
        return loader

    def val_dataloader(self):
        """ Construct the validation data loader """
        loader = None if self.val_set is None else \
            self._dataloader(self.val_set, BucketType.GROUP, self.collate_fn)
        return loader

    def test_dataloader(self):
        """ Construct the test data loader """
        if self.hparams.context_regime == ContextRegime.RANDOM:
            ncontext = int(self.hparams.batch_size * self.hparams.test_context)
            collate_fn = lambda x: collate_sampled_context(x, ncontext=ncontext)
        else:
            collate_fn = self.collate_fn
        loader = None if self.test_set is None else \
            self._dataloader(self.test_set, BucketType.GROUP, collate_fn=collate_fn,
                             fixed_batches=self.test_batches)
        return loader

    @staticmethod
    def add_data_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Add args pertaining to the model and training of the process """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_file", type=str, default="haggling-hbps.h5",
                            help="filename for the haggling data frame (hdf)")
        parser.add_argument("--fold_file", type=str, default="fold0.pkl",
                            help="filename for the train/val group keys (pkl)")
        parser.add_argument("--train_all", default=False, action="store_true",
                            help="use the entire training set ignoring folds")
        parser.add_argument("--bucket_type", type=BucketType,
                            action=EnumAction, default=BucketType.GROUP,
                            help="policy to use for bucketing sequences while batching")
        parser.add_argument("--batch_size", type=int, default=128,
                            help="size of the mini-batch")
        parser.add_argument("--ndata_workers", type=int, default=2,
                            help="number of workers for data loading")
        parser.add_argument("--fix_future_len", default=False,
                            action="store_true",
                            help="discard future sequences shorter than "
                            "specified future length")
        parser.add_argument("--context_regime", type=ContextRegime,
                            action=EnumAction, default=ContextRegime.RANDOM,
                            help="use randomly sampled of fixed context, "
                                 "default RANDOM")
        parser.add_argument("--test_context", type=float, default=0.5,
                            help="fraction of test batch to use as context. "
                                 "only used if context regime is random.")
        parser.add_argument("--feature_set", type=FeatureSet, action=EnumAction,
                            default=FeatureSet.HBPS, help="set of expected features")
        return parser
