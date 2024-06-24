import argparse
import itertools
import logging
from argparse import Namespace
from pathlib import Path
from typing import Union

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger, CSVLogger
from torch.utils.data import DataLoader
import constants.paths as paths
from common.initialization import init_torch
from common.utils import EnumAction, configure_logging
from data.datasets import SyntheticGlancingSameContext, GroupOrderingDataset
from data.loader import collate_sampled_context
from data.types import ModelType, ComponentType, Seq2SeqSamples
from lightning.processes import SPSystemBase
from run.plot_orders import plot_meta_sample
from run.utils import init_model, init_ckpt_callbacks, override_hidden_dims
from data.loader import DataSplit
import torch

# from run.plot_curves import (plot_batch, plot_mixed_context, plot_normal_with_context, plot_z_analysis,
#                              plot_multiple_samples_of_z)


def merge_context(data_split: DataSplit):
    """
    Given a data split, merge the observed and future parts of the context into a single tensor.
    Args:
        data_split: the data split (context and target)

    Returns: a new DataSplit object with the observed and future parts of the context merged into a single tensor
    and unchanged future part of the target

    """
    new_observed = data_split.context.observed.clone()
    new_future = data_split.context.future.clone()

    new_observed = torch.cat([new_observed, new_future], dim=0)
    new_future = torch.zeros(0, new_future.shape[1], new_future.shape[2], new_future.shape[3])

    new_context = Seq2SeqSamples(key=data_split.context.key, observed_start=data_split.context.observed_start,
                                 observed=new_observed, future_len=data_split.context.future_len,
                                 offset=data_split.context.offset, future=new_future)
    return DataSplit(context=new_context, target=data_split.target)


def get_collate_function(to_merge_context, ncontext):
    """
    Returns the collate function for a dataset. Essentially just returns collate_sampled_context lambda. But in case
    merge_context is setup, it wraps this lambda with a lambda that merges the context.
    Args:
        to_merge_context: whether to merge the context future and observed parts into a single observed tensor
        ncontext: the number of samples to put into the context

    Returns: a lambda for collating the context

    """
    if to_merge_context:
        return lambda x: merge_context(collate_sampled_context(x, ncontext=ncontext))
    else:
        return lambda x: collate_sampled_context(x, ncontext=ncontext)


def create_permutations(n: int,
                        observed_length: int,
                        future_length: int,
                        context_size: int,
                        target_size: int,
                        meta_sample_count_per_case: int,
                        single_person_talk_time: int):
    """
    Create a dataset of n people. Creates the n! permutations, where each person talks for `single_person_talk_time`
    timesteps. Then, for each permutation,
    Args:
        n: the number of people in the group (there will be n! total configurations!)
        observed_length: how much of the sequence to put in each observed part
        future_length: how much of the sequence to put in each future part
        context_size: how many (observed, future) pairs to have in the context
        target_size: how many (observed, future) pairs to have in the target
        meta_sample_count_per_case: how many meta samples to create for each permutation
        single_person_talk_time: how many timesteps each person talks for
    Returns: a numpy array of shape (observed_length+future_length, n! * meta_sample_count_per_case, n) with the permutations
    """
    # Create the permutations
    perms = np.array(list(itertools.permutations(range(n))))

    # Create the dataset
    dataset = np.zeros((observed_length + future_length, perms.shape[0] * meta_sample_count_per_case, n))

    # Go over each permutation
    for i, perm in enumerate(perms):

        # Create the full sequence
        seq = np.zeros((n, n))
        for j in range(n):
            seq[j, perm[j]] = 1

        for j in range(meta_sample_count_per_case):
            ind = i * meta_sample_count_per_case + j

            # Choose random starting points for context


#
# def get_custom_dataset_for_validation(args, ncontext):
#     """
#     Returns all data needed for my custom dataset: the train and val loaders,
#     the wave np arrays and train/val Dataset objects for the dataset.
#
#     Args:
#         args: the arguments, defined by the user. In particular, contains the future length,
#         batch size and whether to merge context
#         ncontext: the number of samples to put into the context
#
#     Returns: a tuple with all required loaders, datasets, np arrays
#
#     """
#
#     new_waves = create_my_own_waves(50000)
#     val_cnt = 9000
#     val_waves = new_waves[:, :val_cnt, :]
#     train_waves = new_waves[:, val_cnt:, :]
#
#     train_set = SyntheticGlancingSameContext(train_waves, args.future_len, args.batch_size)
#     val_set = SyntheticGlancingSameContext(val_waves, args.future_len, args.batch_size)
#
#     val_loader = DataLoader(
#         val_set, shuffle=False, batch_size=args.batch_size,
#         collate_fn=(get_collate_function(args.my_merge_context, ncontext))
#     )
#
#     loader = DataLoader(
#         train_set, shuffle=False, batch_size=args.batch_size,
#         collate_fn=(get_collate_function(args.my_merge_context, ncontext))
#     )
#     return loader, val_loader, train_waves, val_waves, train_set, val_set
#

def train(outroot: Path, args: Namespace):
    """
    Train the model on the given training set.

    Args:
        outroot: the output root directory
        args: parameters for training
    """
    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints-synthetic"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Create dataloader and pass to trainer
    people_count = 5
    context_size = 2
    target_size = 1
    args.my_merge_context = False
    meta_sample_count_per_case = 5
    observed_length = 2
    future_length = 2

    args.observed_len = observed_length

    batch_size = context_size + target_size
    full_set = GroupOrderingDataset(
        people_count,
        observed_length=observed_length,
        future_length=future_length,
        context_size=context_size,
        target_size=target_size,
        meta_sample_count_per_case=meta_sample_count_per_case,
    )



    loader = DataLoader(
        full_set, shuffle=False, batch_size=batch_size,
        collate_fn=get_collate_function(args.my_merge_context, context_size),
    )
    #
    for (i, x) in enumerate(loader):
        print("Meta sample is: ")
        print(x)
        permutation = full_set.get_original_permutation(i * batch_size)
        plot_meta_sample(x,
                         permutation,
                         context_size,
                         batch_size)

        if i > 1:
            break


    # # Plot a few samples from training
    # for (i, x) in enumerate(loader):
    #     plot_batch(x.context, x.target)
    #     if i == 2:
    #         break



    # Create trainer
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=None
    )

    # Initialize the model
    process = init_model(args, sp_cls=SPSystemBase)

    trainer.fit(process, train_dataloader=loader,
                val_dataloaders=[])

def test(outroot: Path, args: Namespace, ckpt_path, test_z=False):
    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints-synthetic"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Load the model
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase)

    # Create dataloader and pass to trainer
    people_count = 5
    context_size = 2
    target_size = 1
    args.my_merge_context = False
    meta_sample_count_per_case = 5
    observed_length = 2
    future_length = 2

    args.observed_len = observed_length

    batch_size = context_size + target_size
    full_set = GroupOrderingDataset(
        people_count,
        observed_length=observed_length,
        future_length=future_length,
        context_size=context_size,
        target_size=target_size,
        meta_sample_count_per_case=meta_sample_count_per_case,
    )

    loader = DataLoader(
        full_set, shuffle=False, batch_size=batch_size,
        collate_fn=get_collate_function(args.my_merge_context, context_size),
    )

    # Go over the meta samples
    for (i, meta_sample) in enumerate(loader):
        if i < 3:
            continue
        # Calculate the predictions for the given meta sample
        results = process(meta_sample)
        print("==== ", i)
        print("observed (context): ")
        print(meta_sample.context.observed.shape)
        print(meta_sample.context.observed)

        print("context: ", results.posteriors.q_context.mean, results.posteriors.q_context.scale)
        # print("target: ", results.posteriors.q_target.mean, results.posteriors.q_target.scale)

        permutation = full_set.get_original_permutation(i * batch_size)

        how_many_z_samples = args.nz_samples

        for z_sample in range(how_many_z_samples):

            plot_meta_sample(meta_sample,
                             permutation,
                             context_size,
                             batch_size,
                             results,
                             z_sample)

        if i > 10:
            break

def main() -> None:
    """ Run the main experiment """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-s", "--seed", type=int, default=1234,
                        help="seed for initializing pytorch")
    parser.add_argument("--skip_monitoring", default=False, action="store_true",
                        help="Skip the monitored checkpoint callback")
    parser.add_argument("--model", type=ModelType,
                        action=EnumAction, default=ModelType.SOCIAL_PROCESS,
                        help="type of model to train, default-social process")
    parser.add_argument("--component", type=ComponentType,
                        action=EnumAction, default=ComponentType.RNN,
                        help="type of component modules, default-rnn")
    parser.add_argument("--future_len", type=int, default=20,
                        help="number of future timesteps to predict")
    parser.add_argument("--outdir", type=str,
                        help="root output directory relative to 'artefacts/exp'")
    parser.add_argument("--resdir", type=str, default="results",
                        help="directory to hold result plots")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="size of the mini-batch")
    parser.add_argument("--waves_file", type=str, default="waves.npy",
                        help="filename of the serialized sine data")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level, default: %(default)s)")
    parser.add_argument("--log_file", type=str, default="logs.txt",
                        help="filename for the log file for metrics")
    parser.add_argument("--override_hid_dims", default=False, action="store_true",
                        help="Override representation dimensions to `hid_dim`")
    parser.add_argument("--hid_dim", type=int, default=1024,
                        help="dimension to override representations")
    parser.add_argument("--test", default=False, action="store_true",
                        help="Test instead of train")
    parser.add_argument("--ntest_ctx", type=int, default=785,
                        help="number of test context phase values")
    parser.add_argument("--gen_test_ctx", default=False, action="store_true",
                        help="Generate and serialize context indices for test")
    parser.add_argument("--ckpt_fname", type=str,
                        help="checkpoint filename, expected in "
                             "'outdir'/logs/checkpoints")

    # Arguments added for this experiment
    parser.add_argument("--my_merge_context", default=False, action="store_true", help="Whether to have the context as fully observed, with empty future")
    parser.add_argument("--my_merge_observed_with_future", default=False, action="store_true", help="Whether to merge observed with futures for z encoding")
    parser.add_argument("--my_use_softmax", default=False, action="store_true", help="Whether to use softmax as the last activation function")
    parser.add_argument("--my_ckpt", type=str, help="Checkpoints for my purposes...")
    parser.add_argument("--my_test_z", default=False, action="store_true", help="Whether to take multiple z samples and plot them")

    parser = Trainer.add_argparse_args(parser)

    # Add model specific args
    parser = SPSystemBase.add_model_specific_args(parser)
    args = parser.parse_args()

    # Update arguments
    args.enc_nhid = 32
    args.no_pool = True
    args.nposes = 1

    # Override representations dim if needed
    if args.override_hid_dims:
        args = override_hidden_dims(args, args.hid_dim)

    # Initialize pytorch
    init_torch(args.seed)
    seed_everything(args.seed)

    # Create output directory
    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    args.log_file = str(outroot / args.log_file)

    # Setup logging
    configure_logging(args.log_level, args.log_file)

    # Load data, set paths
    artefacts_dir = (Path(__file__).resolve().parent.parent / "artefacts/")
    dataset_dir = artefacts_dir / "datasets/synthetic/sine-waves"

    if args.test:
        logging.info("Testing")
        # Test
        test(outroot, args, args.my_ckpt, args.my_test_z)

    else:
        # Train
        logging.info("Training")
        train(outroot, args)

if __name__ == "__main__":
    main()

# python -m run.run_ordering_experiment --gpus 1 --future_len 10 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.25 --max_epochs 250 --r_dim 10 --z_dim 5 --override_hid_dims --hid_dim 10 --log_file final_train_log-5.txt --my_use_softmax --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=027.ckpt"

# To see pretty good info:
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 20 --z_dim 10 --override_hid_dims --hid_dim 20 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=178.ckpt"

# With more metasamples trained, kl=0... (does not learn for z.)
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 30 --z_dim 20 --override_hid_dims --hid_dim 30 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=009.ckpt"

# Something that works learning where to go (but bad data for learning q(z|C)).
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 12 --z_dim 8 --override_hid_dims --hid_dim 12 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=019.ckpt"


# Failed experiment: (proper data for learning q(z|C))
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 10 --z_dim 7 --override_hid_dims --hid_dim 10 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=017.ckpt"

# Maybe this can work:
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.25 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1