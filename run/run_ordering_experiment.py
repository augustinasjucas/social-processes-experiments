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
from data.datasets import SyntheticGlancingSameContext, GroupOrderingDataset, DualRandomFFADataset, DualFFADataset, \
    DominatingDataset, FullRandomFFADataset
from data.loader import collate_sampled_context, collate_context_independently
from data.types import ModelType, ComponentType, Seq2SeqSamples
from lightning.processes import SPSystemBase
from run.plot_orders import plot_meta_sample, draw_heatmap, plot_meta_sample_multitime, plot_losses
from run.utils import init_model, init_ckpt_callbacks, override_hidden_dims
from data.loader import DataSplit
import torch

from train.loss import SocialProcessLossCategorical, SocialProcessLoss


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
    # print("==== merging context ====")
    new_observed = data_split.context.observed.clone()
    new_future = data_split.context.future.clone()

    new_observed = torch.cat([new_observed, new_future], dim=0)
    new_future = torch.zeros(0, new_future.shape[1], new_future.shape[2], new_future.shape[3])

    new_context = Seq2SeqSamples(key=data_split.context.key, observed_start=data_split.context.observed_start,
                                 observed=new_observed, future_len=data_split.context.future_len,
                                 offset=data_split.context.offset, future=new_future)
    return DataSplit(context=new_context, target=data_split.target)


def get_collate_function(to_merge_context, ncontext, context_independent_from_target):
    """
    Returns the collate function for a dataset. Essentially just returns collate_sampled_context lambda. But in case
    merge_context is setup, it wraps this lambda with a lambda that merges the context.
    Args:
        to_merge_context: whether to merge the context future and observed parts into a single observed tensor
        ncontext: the number of samples to put into the context

    Returns: a lambda for collating the context

    """
    collate_function = collate_sampled_context
    if context_independent_from_target:
        collate_function = collate_context_independently
    if to_merge_context:
        return lambda x: merge_context(collate_function(x, ncontext=ncontext))
    else:
        return lambda x: collate_function(x, ncontext=ncontext)


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

def get_loaders(args, the_dataset):
    # Create dataloader and pass to trainer
    people_count = args.my_people_count
    context_size = args.my_context_size  # used to be 2
    target_size = args.my_target_size  # used to be 1
    meta_sample_count_per_case = args.my_meta_sample_count_per_case
    observed_length = args.my_observed_length
    future_length = args.my_future_length

    args.observed_len = observed_length

    batch_size = context_size + target_size
    print("batch size = ", batch_size)
    full_set = the_dataset(
        people_count,
        observed_length=observed_length,
        future_length=future_length,
        context_size=context_size,
        target_size=target_size,
        meta_sample_count_per_case=meta_sample_count_per_case,
        repeat_count=args.my_repeat_count,
        how_many_random_permutations=args.my_how_many_random_permutations
    )
    train_size = (int(0.8 * len(full_set)) // batch_size) * batch_size
    val_size = len(full_set) - train_size
    train_dataset = torch.utils.data.Subset(full_set, range(train_size))
    val_dataset = torch.utils.data.Subset(full_set, range(train_size, train_size + val_size))

    train_loader = DataLoader(
        train_dataset, shuffle=False, batch_size=batch_size,
        collate_fn=get_collate_function(args.my_merge_context, context_size, args.my_context_independent_from_target),
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size,
        collate_fn=get_collate_function(args.my_merge_context, context_size, args.my_context_independent_from_target),
    )
    full_loader = DataLoader(
        full_set, shuffle=False, batch_size=batch_size,
        collate_fn=get_collate_function(args.my_merge_context, context_size, args.my_context_independent_from_target),
    )



    return train_loader, val_loader, full_loader, full_set

def train(outroot: Path, args: Namespace, the_dataset):
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

    train_loader, val_loader, _, full_set = get_loaders(args, the_dataset)


    context_size = args.my_context_size
    batch_size = args.my_context_size + args.my_target_size
    # for (i, x) in enumerate(train_loader):
    #
    #     permutation = full_set.get_original_permutation(i * batch_size)
    #     plot_meta_sample_multitime(x, permutation, context_size, batch_size,
    #                                predictions_are_normal=not args.my_predict_categorical)
    #
    #     if i > 1:
    #         break


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

    trainer.fit(process, train_dataloader=train_loader,
                val_dataloaders=[val_loader])

def test(outroot: Path, args: Namespace, ckpt_path, test_z=False, the_dataset=GroupOrderingDataset):
    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints-synthetic"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Load the model
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase)

    # Create dataloader and pass to trainer
    people_count = args.my_people_count
    context_size = args.my_context_size # used to be 2
    target_size = args.my_target_size   # used to be 1
    # args.my_merge_context = False
    meta_sample_count_per_case = args.my_meta_sample_count_per_case
    observed_length = args.my_observed_length
    future_length = args.my_future_length

    args.observed_len = observed_length

    batch_size = context_size + target_size
    full_set = the_dataset(
        people_count,
        observed_length=observed_length,
        future_length=future_length,
        context_size=context_size,
        target_size=target_size,
        meta_sample_count_per_case=meta_sample_count_per_case,
        repeat_count=args.my_repeat_count,
        how_many_random_permutations=args.my_how_many_random_permutations
    )
    train_size = (int(0.8 * len(full_set)) // batch_size) * batch_size
    val_size = len(full_set) - train_size
    train_dataset = torch.utils.data.Subset(full_set, range(train_size))
    val_dataset = torch.utils.data.Subset(full_set, range(train_size, train_size + val_size))

    train_loader = DataLoader(
        train_dataset, shuffle=False, batch_size=batch_size,
        collate_fn=get_collate_function(args.my_merge_context, context_size, args.my_context_independent_from_target),
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size,
        collate_fn=get_collate_function(args.my_merge_context, context_size, args.my_context_independent_from_target),
    )

    loader = train_loader
    if args.my_test_on_validation:
        loader = val_loader

    # Go over the meta samples
    for (i, meta_sample) in enumerate(loader):
        if i < -1:
            continue
        # Calculate the predictions for the given meta sample
        results = [process(meta_sample)]

        # Draw the remaining samples and save them for plotting
        for i in range(args.my_how_many_z_samples_to_plot - 1):
            # Sample from q(z | C)
            z = torch.tensor(results[0].posteriors.q_context.rsample([1]))

            # Initialize the model, denoting the forced z value, and calculate the prediction
            process_new = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=z)
            results.append(process_new(meta_sample))

        if args.my_draw_random_z:
            z = torch.randn_like(results[0].posteriors.q_context.rsample([1]))
            process_new = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=z)
            results.append(process_new(meta_sample))

        permutation = full_set.get_original_permutation(i * batch_size)


        plot_meta_sample_multitime(meta_sample, permutation, context_size, batch_size, results,
                                   final_result_is_random=args.my_draw_random_z,
                                   predictions_are_normal=not args.my_predict_categorical)


        if i > 50:
            break

def test_datasets(outroot: Path, args: Namespace):
    # Define the test datasets
    test_datasets = [DominatingDataset, DualFFADataset]

    # Define the trained models (by path first)
    dual_model_path = args.my_ffa_dual_model
    dual_random_model_path = args.my_ffa_dual_random_model
    full_random_model_path = args.my_ffa_full_random_model
    trained_models = []
    if dual_model_path != "":
        trained_models.append(("Dual", dual_model_path))
    if dual_random_model_path != "":
        trained_models.append(("Dual-random", dual_random_model_path))
    if full_random_model_path != "":
        trained_models.append(("Full-random", full_random_model_path))

    losses = []
    posteriors = []

    # For each model, test it on each dataset, calculate the loss
    for (model_name, model_path) in trained_models:
        model_losses = []
        model_posteriors = []
        for dataset in test_datasets:

            model_dataset_posteriors = []

            # Initialize the model
            process = init_model(args, model_path, sp_cls=SPSystemBase)

            # get the full dataset loader (using get_loaders)
            _, _, full_loader, _ = get_loaders(args, dataset)

            # Evaluate the average loss for the model on the dataset
            total_loss = 0
            total_count = 0
            loss = SocialProcessLossCategorical() if args.my_predict_categorical else SocialProcessLoss()
            for (i, meta_sample) in enumerate(full_loader):
                results = process(meta_sample)
                total_loss += loss.forward(results, meta_sample)[0].detach().numpy()
                total_count += 1

                # Append the posterior normal distribution
                model_dataset_posteriors.append((results.posteriors.q_context.loc.detach().numpy(), results.posteriors.q_context.scale.detach().numpy()))

            model_posteriors.append(model_dataset_posteriors)

            avg_loss = total_loss / total_count
            model_losses.append(avg_loss)
        losses.append(model_losses)
        posteriors.append(model_posteriors)

    # Create model_cnt rows and dataset_cnt columns subplot grid
    plot_losses(losses, posteriors, trained_models, test_datasets, args)

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
    parser.add_argument("--my_plot_z_distribution", default=False, action="store_true", help="Whether to plot q(z | C)")


    parser.add_argument("--my_use_ffa_dual", default=False, action="store_true", help="Whether to generate free-for-all clockwise-anticlockwise samples")
    parser.add_argument("--my_use_dominating", default=False, action="store_true", help="Whether to generate free-for-all clockwise-anticlockwise samples")
    parser.add_argument("--my_use_ffa_dual_random", default=False, action="store_true", help="Whether to generate free-for-all clockwise-anticlockwise samples")
    parser.add_argument("--my_use_ffa_full_random", default=False, action="store_true", help="Whether to generate free-for-all clockwise-anticlockwise samples")
    parser.add_argument("--my_how_many_random_permutations", type=int, default=1)


    parser.add_argument("--my_people_count", type=int, default=5)
    parser.add_argument("--my_context_size", type=int, default=2)
    parser.add_argument("--my_target_size", type=int, default=1)
    parser.add_argument("--my_meta_sample_count_per_case", type=int, default=5)
    parser.add_argument("--my_observed_length", type=int, default=2)
    parser.add_argument("--my_future_length", type=int, default=2)
    parser.add_argument("--my_plot_print_order", default=False, action="store_true", help="Whether to generate free-for-all clockwise-anticlockwise samples")
    parser.add_argument("--my_draw_inferred_edges", default=False, action="store_true")
    parser.add_argument("--my_repeat_count", type=int, default=1, help="How long each speaker talks for")

    parser.add_argument("--my_draw_random_z", default=False, action="store_true")
    parser.add_argument("--my_how_many_z_samples_to_plot", type=int, default=1, help="How long each speaker talks for")

    parser.add_argument("--my_test_on_validation", default=False, action="store_true")
    parser.add_argument("--my_context_independent_from_target", default=False, action="store_true")

    parser.add_argument("--my_predict_categorical", default=False, action="store_true")

    parser.add_argument("--my_analyse_datasets", default=False, action="store_true", help="Whether to plot 'trained-on' vs 'tested-on' matrix losses")
    parser.add_argument("--my_plot_posteriors", default=False, action="store_true", help="Whether to plot 'trained-on' vs 'tested-on' matrix losses")
    parser.add_argument("--my_ffa_dual_model", type=str, default="", help="Path to the dual model, used for dataset analysis")
    parser.add_argument("--my_ffa_dual_random_model", type=str, default="", help="Path to the dual_random model, used for dataset analysis")
    parser.add_argument("--my_ffa_full_random_model", type=str, default="", help="Path to the dual model, used for dataset analysis")

    parser = Trainer.add_argparse_args(parser)

    # Add model specific args
    parser = SPSystemBase.add_model_specific_args(parser)
    args = parser.parse_args()

    the_dataset = GroupOrderingDataset
    if args.my_use_ffa_dual:
        the_dataset = DualFFADataset
    elif args.my_use_dominating:
        the_dataset = DominatingDataset
    elif args.my_use_ffa_dual_random:
        the_dataset = DualRandomFFADataset
    elif args.my_use_ffa_full_random:
        the_dataset = FullRandomFFADataset

    # Update arguments
    # args.enc_nhid = 32
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
    if args.my_analyse_datasets and args.test:
        test_datasets(outroot, args)
    elif args.test:
        logging.info("Testing")
        # Test
        test(outroot, args, args.my_ckpt, test_z=args.my_test_z, the_dataset=the_dataset)

    else:
        # Train
        logging.info("Training")
        train(outroot, args, the_dataset)

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


# === Command for testing the "dominating" setting (3 observed, 1 future):
#  python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 6 --dropout 0.25 --max_epochs 2500 --r_dim 6 --z_dim 5 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 6  --my_context_size 4 --my_target_size 4 --my_observed_length 3 --my_future_length 1 --my_use_dominating --my_meta_sample_count_per_case 50  --my_plot_z_distribution --test --my_repeat_count 2

# Command for training the "random dual" setting: (3 observed, 1 future)
#  python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 6 --dropout 0.25 --max_epochs 2500 --r_dim 6 --z_dim 5 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 6  --my_context_size 4 --my_target_size 4 --my_observed_length 3 --my_future_length 1 --my_use_dominating --my_meta_sample_count_per_case 50  --my_plot_z_distribution --test --my_repeat_count 2



# TRAINED ON FFA DUAL (cant learn FFA dual, z 5-dimensional)
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 6 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=065.ckpt"

# Trained on FFA DUAL (z 1-dimensional, also does not work)
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=039.ckpt --test


# AFTER FIXING BUGS:
# Working on FFA DUAL (z 1-dimensional)
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 1 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=27-monitored_nll=-13.958.ckpt"

# FFA DUAL (z 1-dimensional, 3 context, 4 timesteps)
#  python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 1 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 3 --my_target_size 3 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=031-v0.ckpt --test
# FFA DUAL (z 1-dimensional, 3 context, 4 timesteps, testing on dominating) === CLEARLY, DOES NOT LEARN
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 1 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 3 --my_target_size 3 --my_observed_length 1 --my_future_length 3 --my_use_dominating --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=031-v0.ckpt --test


# FFA DUAL (z 5-dimensional, 3 context, 4 timesteps) -- TOTAL POSTERIOR COLLAPSE!
#  python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 3 --my_target_size 3 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=26-monitored_nll=141.463.ckpt"

# FFA DUAL (z 5-dimensional, 10 context, 10 target, 1 observed, 3 future, 7 people, 200 meta samples per case). I.e., I give a shit-ton of data (WORKS)
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 200  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=014.ckpt
# Same tested on dominating:
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_dominating --my_meta_sample_count_per_case 200  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=014.ckpt


# TRAINING WITH DOMINATING DATASET (LEARNS TO IDENTIFY THE DOMINATING PERSON, but not the clockwise/anticlockwise order!)
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_dominating --my_meta_sample_count_per_case 200  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=64.517.ckpt


# Training with FFA DUAL RANDOM (and testing on it. Z - 5-dimensional LEARNS TO REMEMBER):
# TO BE REDONE! python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --test --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=12-monitored_nll=76.608.ckpt

# training with FFA DUAL RANDOM, z 16-dimensional: (expecting this to learn to remember the context perfectly)
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 16 --z_dim 16 --override_hid_dims --hid_dim 16 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --test --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=19-monitored_nll=109.736.ckpt

# Training with FFA_FULL_RANDOM (q(z|C) collapses, learns to average shit out).
# python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 6 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 6  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --test --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=2-monitored_nll=154.783.ckpt



r"""

Training on dual_ffa
    - Learns dual ffa well
    - Works very bad with dominating. 
        a) 1-dimensional z: based on context decides "clockwise" or "anticlockwise" (or sometimes - interpolated version, probably).
         python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=42-monitored_nll=-14.312.ckpt" --my_test_on_validation
        
        b) 5-dimensional z: Predicts some nonsense, kind of resembling just dual-ffa, but only kinda. Also very innacurate
        python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_dominating --my_meta_sample_count_per_case 200  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --test --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\last-epoch=014.ckpt

Training on dual_ffa_random
    - ATM, does not learn to remember the context, just predicts chess-like pattern
    - Does does not generalize for dominating, pretty much just predicts what it would predict for dual_ffa_random - i.e., chess pattern
    python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 16 --z_dim 16 --override_hid_dims --hid_dim 16 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --test --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=19-monitored_nll=109.736.ckpt
    
    - Does not posterior-collapse
        - Differnet z values result in different end results
        - Given some q(z | C), samples different samples result in very similar output predictions
        - Even then, it still typically predicts chess-like probability distribution, i.e., context seems to be useless
    

Training on full_ffa_random
    - Mayybe learns to remember the context in some cases
    - ??? Generalizes for dominating?
     
     
FULL FFA RANDOM, GOT REMEMBERING TO WORK (KINDA):
python -m run.run_ordering_experiment --gpus 1 --future_len 1 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 1 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 2 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=100.509.ckpt"

FULL FFA RANDOM, BIGGER VERSION, REMEMBERING SEEMS TO KINDA WORK:
 python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 40 --z_dim 40 --override_hid_dims --hid_dim 40 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 2 --my_use_dominating --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 3 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=24-monitored_nll=122.775.ckpt" --my_how_many_z_samples_to_plot 4

Trying to get the remembering to work on longer datasets (with larger repeat count) SEEMS TO BE JUST DOING COUNTING:
 python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 40 --z_dim 40 --override_hid_dims --hid_dim 40 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 4 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=25-monitored_nll=124.882.ckpt"

INDEPENDENT INFORMATION PROOF:
python -m run.run_ordering_experiment --gpus 1 --future_len 1 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 3 --my_observed_length 1 --my_future_length 1 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 2 --my_context_independent_from_target --test --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=4-monitored_nll=169.545.ckpt --my_test_on_validation

TRAINING DUAL FFA RANDOM ON THE SAME ARCHITECTURE AS fulll ramdom that learns; (does not really work!)
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 40 --z_dim 40 --override_hid_dims --hid_dim 40 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --nlayers 3 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=21-monitored_nll=86.033.ckpt"







USING CATEGORICAL CROSS-ENTROPY (NO SAMPLING FROM A CATEGORICAL DISTRIBUTION)
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical  --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=30-monitored_nll=0.035.ckpt"


USING CATEGORICAL DISTRIBUTION:

dual:
z 1-dimensional: python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical  --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=0.038.ckpt"
z 5-dimensional:  python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 8 --z_dim 5 --override_hid_dims --hid_dim 8 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual --my_meta_sample_count_per_case 200  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=33-monitored_nll=0.877.ckpt"

dual-random:
 python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 16 --z_dim 16 --override_hid_dims --hid_dim 16 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_dual_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --my_predict_categorical --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=17-monitored_nll=1.069.ckpt"

FFA full random:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 6 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 6  --my_context_size 10 --my_target_size 10 --my_observed_length 1 --my_future_length 3 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 2000 --my_predict_categorical --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=6-monitored_nll=1.797.ckpt"

Categorical small full random:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 40 --z_dim 40 --override_hid_dims --hid_dim 40 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_dominating --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 3 --my_predict_categorical --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=5-monitored_nll=1.386.ckpt" --test --my_how_many_z_samples_to_plot 4


======== (controlled setting: 1-dimensional z) ========
Trained on DUAL dataset:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical  --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=0.038.ckpt"

Trained on DUAL-RANDOM dataset:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual_random --my_meta_sample_count_per_case 100  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=191-monitored_nll=0.900.ckpt" 

Trained on FULL-RANDOM dataset:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=8-monitored_nll=1.984.ckpt"

Displays all of those:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 7 --dropout 0.0 --max_epochs 2500 --r_dim 6 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 7  --my_context_size 4 --my_target_size 4 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --test --my_analyse_datasets --my_ffa_dual_model  "out-3\logs\checkpoints-synthetic\mon-epoch=29-monitored_nll=0.038.ckpt" --my_ffa_dual_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=191-monitored_nll=0.900.ckpt" --my_ffa_full_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=8-monitored_nll=1.984.ckpt" --my_plot_posteriors


remembering with lower z (:
python -m run.run_ordering_experiment --gpus 1 --future_len 1 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 16 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_use_softmax --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 1 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 4 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=42-monitored_nll=93.964.ckpt"

remembering with lower z:
    just same with "categorical" changed (context remembering works, but does not use X): python -m run.run_ordering_experiment --gpus 1 --future_len 1 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_predict_categorical --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 1 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 2 --test --my_ckpt out-3\logs\checkpoints-synthetic\mon-epoch=58-monitored_nll=7.509.ckpt
    with higher nlayers=4: python -m run.run_ordering_experiment --gpus 1 --future_len 1 --outdir out-3 --skip_normalize_rot --data_dim 4 --dropout 0.0 --max_epochs 2500 --r_dim 32 --z_dim 32 --override_hid_dims --hid_dim 32 --log_file final_train_log-5.txt --my_predict_categorical --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 4  --my_context_size 5 --my_target_size 1 --my_observed_length 1 --my_future_length 1 --my_use_ffa_full_random --my_meta_sample_count_per_case 1  --my_plot_z_distribution --my_repeat_count 1 --my_merge_observed_with_future --my_merge_context --my_how_many_random_permutations 4000 --nlayers 4 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=20-monitored_nll=7.533.ckpt"
 
 



====== MOST REALISTIC SETTING: 5 people, 8 context interactions, 3 target interactions, 
1-dimensional z:
DUAL:        python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 1 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --test --my_ckpt  "out-3\logs\checkpoints-synthetic\mon-epoch=12-monitored_nll=0.006.ckpt"
DUAL-RANDOM: python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 1 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual_random --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=62-monitored_nll=38.022.ckpt"
FULL-RANDOM: python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 1 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random  --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --test --my_ckpt  "out-3\logs\checkpoints-synthetic\mon-epoch=22-monitored_nll=70.348.ckpt"
 displays all:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --test --my_analyse_datasets --my_ffa_dual_model "out-3\logs\checkpoints-synthetic\mon-epoch=12-monitored_nll=0.006.ckpt" --my_ffa_dual_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=62-monitored_nll=38.022.ckpt" --my_ffa_full_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=22-monitored_nll=70.348.ckpt" --my_plot_posteriors

 
multidimensional z:
DUAL:        python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-6.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --test --my_ckpt "out-3\logs\checkpoints-synthetic\mon-epoch=28-monitored_nll=0.001.ckpt"
DUAL-RANDOM: python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-7.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_dual_random --my_meta_sample_count_per_case 100 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --test --my_ckpt C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=103-monitored_nll=38.794-v0.ckpt
FULL-RANDOM: python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-8.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random  --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000 --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --nlayers 5 --test --my_ckpt  "out-3\logs\checkpoints-synthetic\mon-epoch=76-monitored_nll=69.785.ckpt"
 displays all:
python -m run.run_ordering_experiment --gpus 1 --future_len 2 --outdir out-3 --skip_normalize_rot --data_dim 5 --dropout 0.0 --max_epochs 2500 --r_dim 64 --z_dim 64 --override_hid_dims --hid_dim 64 --log_file final_train_log-5.txt --fix_variance --nz_samples 1 --my_plot_print_order --my_people_count 5  --my_context_size 8 --my_target_size 3 --my_observed_length 1 --my_future_length 2 --my_use_ffa_full_random --my_meta_sample_count_per_case 1 --my_how_many_random_permutations 2000  --my_plot_z_distribution --my_repeat_count 2 --my_merge_observed_with_future --my_merge_context --my_predict_categorical --test --my_analyse_datasets --my_ffa_dual_model "out-3\logs\checkpoints-synthetic\mon-epoch=28-monitored_nll=0.001.ckpt" --my_ffa_dual_random_model "C:\Users\augus\Desktop\social-processes-experiments\out-3\logs\checkpoints-synthetic\mon-epoch=103-monitored_nll=38.794-v0.ckpt" --my_ffa_full_random_model "out-3\logs\checkpoints-synthetic\mon-epoch=76-monitored_nll=69.785.ckpt" --my_plot_posteriors


 installing pip:
 pip install numpy==1.26.4
"""