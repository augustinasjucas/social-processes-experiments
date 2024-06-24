#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: sine_dataset.py
# Created Date: Saturday, November 16th 2019, 1:20:51 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import abc
import itertools
import logging
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from functools import cached_property
from itertools import chain
from typing import Any, Callable, List, NamedTuple, OrderedDict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils import data

from .types import Seq2SeqSamples, BucketType


class ToySine(data.Dataset):

    """ Toy dataset with frequency swept sine waves. """

    def __init__(self, waves: np.ndarray, future_len: int) -> None:
        """ Initialise the dataset.

        Args:
            waves       : The numpy data array of shape
                          (seq_length, n_samples, data_dim)
            future_len  : The number of time steps for prediction

        """
        assert future_len < waves.shape[0]
        self.waves = waves
        self.future_len = future_len
        self.seq_len = waves.shape[0]

    def __getitem__(self, idx: int) -> Seq2SeqSamples:
        """ Returns the observed, pred sequence corresponding to the index idx

        Each array in the returned sample is of shape (seq_len, 1, data_dim)
        The second dimension is to simulate the people dimension in real data
        """
        # HACK TODO: Remove key dependency
        return Seq2SeqSamples(
            key=(idx, idx), observed_start=0,
            observed=self.waves[:self.seq_len-self.future_len, idx][:, np.newaxis,:],
            future_len=self.future_len, offset=1,
            future=self.waves[self.seq_len-self.future_len:, idx][:, np.newaxis,:])

    def __len__(self) -> int:
        return self.waves.shape[1]


class GroupSequence(NamedTuple):

    """ Encapsulate start, end, and features for a sequence """

    start: int
    end: int
    features: Any


class SequencePair(NamedTuple):

    """ Local structure to encapsulate a sequence pair """

    key: Tuple[int, int]
    obs_idx: int
    fut_idx: int
    obs_len: int
    fut_len: int
    offset: int


# Type aliases for bucket maps. See `compute_group_buckets` and `compute_sequence_buckets` for details
GroupBucketMap = OrderedDict[Tuple[int, int, int], OrderedDict[int, List]]
SeqBucketMap = OrderedDict[Tuple[int, int], OrderedDict[int, List]]
BucketMap = Union[GroupBucketMap, SeqBucketMap]


def _compute_buckets(pairs: Sequence[SequencePair], key_extractor: Callable) -> BucketMap:
    """ Helper function to compute bucket maps

    Args:
        pairs           --  The list of SequencePair objects to bucket
        key_extractor   --  A function that returns a key given a SequencePair

    Returns a BucketMap object (doesn't do any explicit type checking)
    """
    bucket_map = OrderedDict()
    for idx, pair in enumerate(pairs):
        key = key_extractor(pair)
        if key not in bucket_map:
            # Create a new dictionary for the key and add
            obs_map = OrderedDict()
            bucket_map[key] = obs_map
        else:
            # Access the map for the key
            obs_map = bucket_map[key]
        # Add the idx to the appropriate observed idx list
        if pair.obs_idx not in obs_map:
            obs_map[pair.obs_idx] = [idx]
        else:
            obs_map[pair.obs_idx].append(idx)
    return bucket_map


def compute_group_buckets(pairs: List[SequencePair]) -> GroupBucketMap:
    """ Compute the bucket map for the sequences

    Map a tuple of (group_id, obs_len, fut_len) to a map of observed index to
    list of indices
    """
    key_extractor = lambda pair: (pair.key[0], pair.obs_len, pair.fut_len)
    return _compute_buckets(pairs, key_extractor)


def compute_seq_buckets(pairs: List[SequencePair]) -> SeqBucketMap:
    """ Compute the bucket map for the sequences

    Map a tuple of (obs_len, fut_len) to a map of observed index to
    list of indices
    """
    key_extractor = lambda pair: (pair.obs_len, pair.fut_len)
    return _compute_buckets(pairs, key_extractor)


class SocialDatasetInterface(data.Dataset, metaclass=abc.ABCMeta):
    """ Abstract interface for a SocialDataset """
    @property
    @abc.abstractmethod
    def group_bucket_map(self) -> GroupBucketMap:
        """ Return a GroupBucketMap for aiding generation of batches """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def seq_bucket_map(self) -> SeqBucketMap:
        """ Return a SeqBucketMap for aiding generation of batches """
        raise NotImplementedError


# Map for accessing bucket attribute name by type
BUCKET_ATTR_NAME = {BucketType.GROUP: "group_bucket_map", BucketType.SEQ: "seq_bucket_map"}


class SocialDataset(SocialDatasetInterface):

    """ Encapsulate the synthetic social dataset created in Blender3D """

    @staticmethod
    def group_fields() -> List:
        return ["group_id", "group_size"]

    def __init__(self, obs_df: pd.DataFrame, hparams: Namespace,
                 feature_fields: List, fut_df: pd.DataFrame = None) -> None:
        """ Initialize the dataset object

        Args:
            obs_df          --  Dataframe consisting of behavioral data for
                                groups comprising observed sequences
            hparams         --  Parameters to process the data.
                                Refer `add_dataset_specific_args`
            feature_fields  --  The list of column names for the features
            fut_df          --  Dataframe consisting of behavioral data for
                                groups comprising future sequences

        """
        assert (0 <= hparams.overlap < 1), (
            "overlap must be between 0 inclusive and 1 exclusive"
        )
        self.obs_len = hparams.observed_len
        self.future_len = hparams.future_len
        self.hparams = hparams
        self.obs_df = obs_df
        self.fut_df = obs_df if fut_df is None else fut_df
        assert (not self.obs_df.isnull().values.any()) and (not self.fut_df.isnull().values.any()), \
            "SocialDataset init; Either the observed or future dataframe has NAN values!"
        self.feature_fields = feature_fields
        # Compute the keys and make sure they are the same for observed
        # and future dfs
        obs_keys = obs_df.groupby(SocialDataset.group_fields()).groups.keys()
        fut_keys = self.fut_df.groupby(SocialDataset.group_fields()).groups.keys()
        assert (obs_keys == fut_keys), ("dataframes must contain same groups")
        self.group_keys = obs_keys
        # Map group keys to (observed sequences, future_seqs)
        self.group_seqs = None
        self.pairs = []

    def _construct_pairs(
            self, group_seqs: OrderedDict, fix_future_len=False
        ) -> OrderedDict[Tuple[str, int], SequencePair]:
        """ Construct observed, future pairs from all sequences

        Args:
            group_seqs      --  the OrderedDict mapping group keys to
                                (observed sequences, future_seqs) where the
                                sequences are of type List[GroupSequence]
            fix_future_len  --  restrict future sequences to future_len if True

        Returns
            An ordered dict mapping group key to the list of SequencePair
            objects for that group

        """
        logging.info("[*] Constructing observed-future pairs")
        stride = self.hparams.time_stride
        group_pairs = OrderedDict()
        for key in self.group_keys:
            logging.debug("Constructing seq pairs for group: {}".format(key))
            s_pairs = []
            obs_seqs, fut_seqs = group_seqs[key]
            logging.debug(f"Number of unique seqs for key {key}: obs - "
                         f"{len(obs_seqs)}, fut - {len(fut_seqs)}")
            for i, s_obs in enumerate(obs_seqs):
                for j, s_fut in enumerate(fut_seqs):
                    s_obslen = ((s_obs.end - s_obs.start) // stride) + 1
                    s_futlen = ((s_fut.end - s_fut.start) // stride) + 1
                    offset = s_fut.start - s_obs.end
                    logging.debug(
                        "OBS [{}-{}; len-{}], FUT [{}-{}; len-{}]".format(
                            s_obs.start, s_obs.end, s_obslen,
                            s_fut.start, s_fut.end, s_futlen)
                    )
                    # Add pair of obs sequence is of length self.obs_len
                    # and the future sequence starts after obs seq ends
                    predicate = (
                        (s_obslen == self.obs_len)
                        and (0 < offset <= self.hparams.max_future_offset)
                        and (s_futlen == self.future_len or not fix_future_len)
                        # Return true for the last check when filter_futures
                        # is False.
                    )
                    if predicate:
                        logging.debug(
                            "Adding pair: "
                            "OBS [{}-{}; len-{}], FUT [{}-{}; len-{}]".format(
                                s_obs.start, s_obs.end, s_obslen,
                                s_fut.start, s_fut.end, s_futlen)
                        )
                        s_pairs.append(
                            SequencePair(key, i, j, s_obslen, s_futlen, offset)
                        )
            group_pairs[key] = s_pairs
        return group_pairs

    def _sequences_for_group(
            self, key: Tuple[int, int], df: pd.DataFrame,
            seq_len: int, overlap: float
        ) -> List[GroupSequence]:
        """ Compute samples for a unique group (assumes contiguous chunk of data) """
        # key is a tuple of (group_id, group_size)
        _, gsize = key
        nrows = np.arange(len(df))
        # Minimum step size is equal to 1 timestep or gsize rows
        step_size = max(int(np.rint(((1 - overlap) * seq_len) * gsize)), gsize)
        nseq_rows = seq_len * gsize
        starts = nrows[::step_size]
        bounds = list(zip(starts, starts+nseq_rows))
        s_data = lambda b: df.iloc[b[0]:b[1]]
        # A sequence comprises of start_frame, end_frame, and data
        seqs = []
        for s in map(s_data, bounds):
            if len(s) >= nseq_rows: # Check there are sufficient rows
                values = s[self.feature_fields].values.astype(np.float32)
                values = values.reshape(-1, gsize, values.shape[-1])
                gseq = GroupSequence(
                    s["frame"].iloc[0], s["frame"].iloc[-1], values
                )
                seqs.append(gseq)
        return seqs

    def _compute_samples_for_df(
            self, df: pd.DataFrame, seq_len: int, overlap: float
        ) -> List[GroupSequence]:
        """ Compute sequences for specific dataframe """
        stride = self.hparams.time_stride
        # Sample every `stride` frames, starting from the frame in the
        # first row of the dataframe
        dfs = df[(df.frame - int(df.frame.iloc[0])) % stride == 0]
        seq_dict = OrderedDict()
        for g in dfs.groupby(SocialDataset.group_fields()):
            key, group_df = g
            seqs_for_group = []
            # Process contiguous chunks
            predicate = (group_df.frame - group_df.frame.shift() > stride)
            for _, chunk in group_df.groupby(predicate.cumsum()):
                seqs = self._sequences_for_group(key, chunk, seq_len, overlap)
                seqs_for_group.extend(seqs)
            seq_dict[key] = seqs_for_group
        return seq_dict

    def compute_samples(self, fix_future_len=False) -> None:
        """ Constructs the sample sequences from the dataframe """
        overlap = self.hparams.overlap
        obs_dict = self._compute_samples_for_df(self.obs_df, self.obs_len, overlap)
        overlap = 1 if self.hparams.all_futures else overlap
        fut_dict = self._compute_samples_for_df(self.fut_df, self.future_len, overlap)
        self.group_seqs = OrderedDict(
            {k:(obs_dict.get(k), fut_dict.get(k)) for k in self.group_keys}
        )
        # Construct (obs, future) pairs for each group
        group_pairs = self._construct_pairs(self.group_seqs, fix_future_len)
        self.pairs = list(chain(*group_pairs.values()))

    @cached_property
    def group_bucket_map(self) -> GroupBucketMap:
        """ Override SocialDatasetInterface.group_bucket_map()

        Map a tuple of (group_id, obs_len, fut_len) to a map of observed idx to sample indices
        """
        return compute_group_buckets(self.pairs)

    @cached_property
    def seq_bucket_map(self) -> SeqBucketMap:
        """ Override SocialDatasetInterface.seq_bucket_map()

        Map a tuple of (obs_len, fut_len) to a map of observed idx to sample indices
        """
        return compute_seq_buckets(self.pairs)

    def _convert_pair(self, pair: SequencePair, seqs: OrderedDict) -> Seq2SeqSamples:
        """ Convert a SequencePair to Seq2SeqSamples """
        # key is a tuple of (group_id, group_size)
        obs = seqs[pair.key][0][pair.obs_idx]
        fut = seqs[pair.key][1][pair.fut_idx]

        return Seq2SeqSamples(
            key=pair.key,
            observed_start=obs.start,
            observed=obs.features,
            future_len=fut.features.shape[0],
            offset=pair.offset,
            future=fut.features
        )

    def __getitem__(self, idx: int) -> Seq2SeqSamples:
        """ Returns Seq2SeqSamples for a single group

        Each tensor in the sample is of shape (seq_len, npeople, data_dim)

        """
        pair = self.pairs[idx]
        return self._convert_pair(pair, self.group_seqs)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Add args pertaining to the model and training of the process """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--observed_len", type=int, default=10,
                            help="number of observed timesteps (# rows from the dataframe)")
        parser.add_argument("--future_len", type=int, default=10,
                            help="number of future timesteps to predict (# rows from the dataframe)")
        parser.add_argument("--time_stride", type=int, default=1,
                            help="sampling rate of rows in the dataset, expressed as multiple of frame difference "
                                 "between rows; for eg."
                                 "for a dataset where each row is 20 frames apart, a stride of 60 means every third "
                                 "row will be taken.")
        parser.add_argument("--overlap", type=float, default=0.8,
                            help="Overlap between observed sequences [0, 1)")
        parser.add_argument("--all_futures", default=False, action="store_true",
                            help="Take all future sequences within max offset "
                                 "instead of applying overlap")
        parser.add_argument("--max_future_offset", type=int, default=150,
                            help="maximum offset in frame values between the end of the observed "
                            "and beginning of the future sequence, inclusive")

        return parser


class SocialUnpairedContextDataset(SocialDataset):

    """ Compute seq2seq samples with past sequences as context

    The context sequences are not split into observed and future pairs. This
    is meant to be used for the SocialProcess model where the context future
    sequences are not used for encoding the latent representations. Hence, only
    unique context sequences are required. In the case where multiple futures
    are possible for a given observed sequence, splitting the context into
    (observed, future) pairs would result in duplicate observed sequences which
    is not desirable in this case. Refer `SocialPairedContextDataset` if
    (observed, future) pairs for the context sequences is desirable.

    """

    def __init__(
            self, obs_df: pd.DataFrame, ctx_df: pd.DataFrame,
            hparams: Namespace, feature_fields: List,
            fut_df: pd.DataFrame = None
        ) -> None:
        """ Initialize the dataset object

        Args:
            obs_df          --  Dataframe consisting of behavioral data for
                                groups comprising observed sequences for which
                                to predict the future
            ctx_df          --  Dataframe consisting of past behavioral data
                                for groups comprising the context at evaluation
            hparams         --  Parameters to process the data.
                                Refer `add_dataset_specific_args`
            feature_fields  --  The list of column names for the features
            fut_df          --  Dataframe consisting of behavioral data for
                                groups comprising future sequences

        """
        super().__init__(obs_df, hparams, feature_fields, fut_df)
        self.ctx_df = ctx_df
        ctx_keys = ctx_df.groupby(SocialDataset.group_fields()).groups.keys()
        assert (ctx_keys == self.group_keys), (
            "dataframes must contain same groups"
        )
        # Map group ids to list of context seqeuences
        self.ctx_data = OrderedDict()

    def compute_samples(self, fix_future_len=False) -> None:
        """ Compute observed, future target pairs and context sequences """
        ctx_seqs = self._compute_samples_for_df(self.ctx_df, self.obs_len,
                                                self.hparams.overlap)
        for key in ctx_seqs:
            # Filter context sequences to make sure they're of the same length
            seqs = ctx_seqs[key]
            filtered = [s for s in seqs if s.features.shape[0] ==  self.obs_len]
            ctx_seq_data = [s.features for s in filtered]
            ctx = np.stack(ctx_seq_data, axis=1)
            self.ctx_data[key] = ctx
        super().compute_samples(fix_future_len=fix_future_len)

    def __getitem__(self, idx: int) -> Tuple[Seq2SeqSamples, np.ndarray]:
        """ Returns Seq2SeqSamples and corresponding context sequences

        Each tensor in the sample is of shape (seq_len, group_size, data_dim)

        Returns:
            A tuple of the Seq2Seq samples and context data of shape
            (seq_len, ctx_size, group_size, data_dim)
        """
        samples = super().__getitem__(idx)
        context = self.ctx_data[samples.key]
        return (samples, context)


class SocialPairedContextDataset(SocialDataset):

    """ Compute evaluation seq2seq samples with past sequences as context

    The context sequences are split into observed and future pairs. Refer to
    `SocialEvalDataset` for a discussion about this.

    """

    def __init__(
            self, obs_df: pd.DataFrame, ctx_df: pd.DataFrame,
            hparams: Namespace, feature_fields: List,
            fut_df: pd.DataFrame = None
        ) -> None:
        """ Initialize the dataset object

        Args:
            obs_df          --  Dataframe consisting of behavioral data for
                                groups comprising observed sequences for which
                                to predict the future
            ctx_df          --  Dataframe consisting of past behavioral data
                                for groups comprising the context at evaluation
            hparams         --  Parameters to process the data.
                                Refer `add_dataset_specific_args`
            feature_fields  --  The list of column names for the features
            fut_df          --  Dataframe consisting of behavioral data for
                                groups comprising future sequences

        """
        super().__init__(obs_df, hparams, feature_fields, fut_df)
        self.ctx_df = ctx_df
        ctx_keys = ctx_df.groupby(SocialDataset.group_fields()).groups.keys()
        assert (ctx_keys == self.group_keys), (
            "dataframes must contain same groups"
        )
        # Map group ids to a single Seq2SeqSamples of all context sequences
        # for that group
        self.group_ctx = OrderedDict()

    def compute_samples(self, fix_future_len=False) -> None:
        """ Compute observed, future target pairs and context sequences """
        # Compute context sequences
        overlap = self.hparams.overlap
        ctx_obs = self._compute_samples_for_df(self.ctx_df, self.obs_len, overlap)
        overlap = 1 if self.hparams.all_futures else overlap
        ctx_fut = self._compute_samples_for_df(self.ctx_df, self.future_len, overlap)
        ctx_seqs = OrderedDict(
            {k:(ctx_obs.get(k), ctx_fut.get(k)) for k in self.group_keys}
        )
        # Construct context (obs, future) pairs for each group
        # ctx_pairs is an ordered dict of group key to list of SequencePair
        # ensuring all observed and future sequences are of same length
        ctx_pairs = self._construct_pairs(ctx_seqs, fix_future_len=True)
        # Collate all context pairs into a single Seq2SeqSample for each group
        for key in ctx_pairs:
            samples = [self._convert_pair(p, ctx_seqs) for p in ctx_pairs[key]]
            self.group_ctx[key] = samples
        super().compute_samples(fix_future_len=fix_future_len)

    def __getitem__(self, idx: int) -> Tuple[Seq2SeqSamples, List[Seq2SeqSamples]]:
        """ Returns Seq2SeqSamples and corresponding context sequences

        Each tensor in the sample is of shape (seq_len, group_size, data_dim)

        Returns:
            A tuple of the target and context Seq2Seq samples.
            The sequence of context observed and future pairs

        """
        samples = super().__getitem__(idx)
        context = self.group_ctx[samples.key]
        return (samples, context)


class SyntheticGlancingSameContext(data.Dataset):

    """ Dataset for the synthetic glancing experiment, where the context consists of the same type of curves (either clamped or not)"""

    def __init__(self, arr: np.ndarray, future_len: int, batch_size: int, mixed_context: bool = False) -> None:
        """Initializes the dataset. Does not compute anything.

        We do some cheating here. I.e., the dataset is aware of the batch_size before hand (which it probably should not be
        given). This allows the dataset to put all the sequences of the same type together in a single block in the dataset.
        Then the dataloader will not do any shuffling (that is done here), and as a result, get all the sequences of the same
        type together in a single batch.

        Args:
            arr (np.ndarray): the array containing the data
            future_len (int): the length of the future sequence
            batch_size (int): the batch size, i.e., the number of sequences in a meta-sample
            mixed_context (bool): whether to shuffle all of the sequences. I.e., it forces actually using mixed context,
                                  which is not really the point of this dataset. But it is useful for some experiments.
        """
        # Create the array that stores the indices
        arr_indices = np.arange(arr.shape[1])

        # Take every second element in the sequence and copy them as well
        self.first_arr = arr[:, ::2, :]
        self.second_arr = arr[:, 1::2, :]
        first_arr_indices = arr_indices[::2]
        second_arr_indices = arr_indices[1::2]

        # Make a deep copy of first and second arrays and shuffle them
        shuf1 = np.random.permutation(self.second_arr.shape[1])
        shuf2 = np.random.permutation(self.first_arr.shape[1])

        self.first_arr = self.first_arr.copy()
        self.first_arr = self.first_arr[:, shuf1, :]

        self.second_arr = self.second_arr.copy()
        self.second_arr = self.second_arr[:, shuf2, :]

        first_arr_indices = first_arr_indices[shuf1]
        second_arr_indices = second_arr_indices[shuf2]

        # Resize the arrays to be divisible by batch size (so drop a few curves, but whatever)
        self.first_arr = self.first_arr[:, :(self.first_arr.shape[1] // batch_size) * batch_size, :]
        self.second_arr = self.second_arr[:, :(self.second_arr.shape[1] // batch_size) * batch_size, :]

        first_arr_indices = first_arr_indices[:(first_arr_indices.shape[0] // batch_size) * batch_size]
        second_arr_indices = second_arr_indices[:(second_arr_indices.shape[0] // batch_size) * batch_size]

        # Split them up into batches
        first_batches = np.split(self.first_arr, self.first_arr.shape[1] // batch_size, axis=1)
        second_batches = np.split(self.second_arr, self.second_arr.shape[1] // batch_size, axis=1)

        first_batches_indices = np.split(first_arr_indices, first_arr_indices.shape[0] // batch_size)
        second_batches_indices = np.split(second_arr_indices, second_arr_indices.shape[0] // batch_size)

        # Concatenate the lists as python list
        self.arr = first_batches + second_batches
        self.arr_indices = first_batches_indices + second_batches_indices

        # Shuffle the list
        shuf3 = np.random.permutation(len(self.arr))
        self.arr = [self.arr[i] for i in shuf3]
        self.arr_indices = [self.arr_indices[i] for i in shuf3]

        # Concatenate the list back to a numpy array
        self.arr = np.concatenate(self.arr, axis=1)
        self.arr_indices = np.concatenate(self.arr_indices)

        # In case we actually want to use a mixed context, just drop what we did and shuffle everything
        if mixed_context:
            shuf4 = np.random.permutation(self.arr.shape[1])
            self.arr = self.arr[:, shuf4, :]
            self.arr_indices = self.arr_indices[shuf4]

        self.future_len = future_len
        self.batch_size = batch_size
        self.taken_cnt = 0

    def get_index(self, idx: int) -> int:
        """
        Get the corresponding index in the original np file, of some sequence from the dataset.
        I.e., this is useful when we will want to get the "complement" of a sequence, since it is
        stored in the original np file.
        Args:
            idx: the index in *this* dataset

        Returns: the corresponding index in the *original* np file

        """
        return self.arr_indices[idx]

    def __getitem__(self, idx: int) -> Seq2SeqSamples:
        """ Returns the requested sample.
        """
        obs = self.arr[:-self.future_len, idx, :]
        fut = self.arr[-self.future_len:, idx, :]

        # reshape to [len; 1; 1]
        obs = obs[:, np.newaxis, :]
        fut = fut[:, np.newaxis, :]
        return Seq2SeqSamples(future_len=self.future_len, key=[idx // 2], observed=obs, future=fut, observed_start=0)

    def __len__(self) -> int:
        return self.arr.shape[1]

class GroupOrderingDataset(data.Dataset):
    def __init__(self,
                 n: int,
                 observed_length: int,
                 future_length: int,
                 context_size: int,
                 target_size: int,
                 meta_sample_count_per_case: int) -> None:
        """
        Create a dataset of n people. Creates the n! permutations, where each person talks for `single_person_talk_time`
        timesteps.
        Produces a list of Seq2SeqSamples, where each sample is an observed-future pair. Makes sure that each consecutive
        (target_size+context_size) correspond to the same metasample.
        Args:
            n: the number of people in the group (there will be n! total configurations!)
            observed_length: how much of the sequence to put in each observed part
            future_length: how much of the sequence to put in each future part
            context_size: how many (observed, future) pairs to have in the context
            target_size: how many (observed, future) pairs to have in the target
            meta_sample_count_per_case: how many meta samples to create for each permutation

        """

        self.meta_samples_per_case = meta_sample_count_per_case
        self.context_size = context_size
        self.target_size = target_size

        # Create an rng
        rng = np.random.default_rng(seed=0)

        # Create the permutations
        perms = np.array(list(itertools.permutations(range(n))))

        # Save the original permutations
        self.original_permutations = perms


        # Transform the permutations into 1-hot encoded sequences (shape [n, n]), where first dimension is time dimension
        perms = np.eye(n)[perms]

        # At the moment, perms has shape [perm_count (batch size), permutation_length, n], transpose to [permutation_length, perm_count (batch size), n]
        perms = np.transpose(perms, (1, 0, 2))

        # Repeat the same permutation twice in the list
        perms = np.concatenate([perms, perms], axis=0)

        # Shuffle the permutations (according to 1st axis) and shuffle the original_permutations accordingly
        shuf = rng.permutation(perms.shape[1])
        perms = perms[:, shuf, :]
        self.original_permutations = self.original_permutations[shuf]

        # Create a list for storing sequences
        self.sequences = []

        # Create a list for storing a list of sequences for each meta sample
        temp_sequences = []
        self.corresponding_original_permutations = []

        # Go over each permutation
        for i in range(perms.shape[1]):
            perm = perms[:, i, :].astype(np.float32)

            for j in range(meta_sample_count_per_case):
                ind = i * meta_sample_count_per_case + j

                # Choose random starting points for context and target
                start_indices = rng.integers(0, n, context_size+target_size)

                meta_sample = []

                # Extract (context_size+target_size) number of subsequences of length (observed_length+future_length)
                for k in range(context_size+target_size):

                    # Extract the observed part
                    observed_part = perm[start_indices[k]:start_indices[k]+observed_length, :]

                    # Extract the future part
                    future_part = perm[start_indices[k]+observed_length:start_indices[k]+observed_length+future_length, :]

                    # Create a sample
                    sample = Seq2SeqSamples(
                        future_len=future_length,
                        key=[ind],
                        observed=observed_part[:, np.newaxis, :],
                        future=future_part[:, np.newaxis, :],
                        observed_start=start_indices[k]
                    )

                    # Append the sample to the list
                    meta_sample.append(sample)

                temp_sequences.append(meta_sample)
                self.corresponding_original_permutations.append(i)

        # Shuffle the temp sequences
        shuf = rng.permutation(len(temp_sequences))
        temp_sequences = [temp_sequences[i] for i in shuf]
        self.corresponding_original_permutations = np.array(self.corresponding_original_permutations)[shuf]

        # Flatmap the temp_sequences into self.sequences
        self.sequences = list(itertools.chain(*temp_sequences))


    def get_original_permutation(self, idx: int):
        """
        Get the original permutation that was used to create the idx-th sample in the dataset.
        Args:
            idx: the index of the sample in the dataset

        Returns: the original permutation that was used to create the idx-th sample in the dataset

        """
        return self.original_permutations[self.corresponding_original_permutations[idx // (self.context_size + self.target_size)]]

    def __getitem__(self, idx: int) -> Seq2SeqSamples:
        """ Returns the requested sample.
        """
        return self.sequences[idx]

    def __len__(self) -> int:
        return len(self.sequences)
