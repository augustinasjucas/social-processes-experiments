import argparse
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
from data.datasets import SyntheticGlancingSameContext
from data.loader import collate_sampled_context
from data.types import ModelType, ComponentType, Seq2SeqSamples
from lightning.processes import SPSystemBase
from run.utils import init_model, init_ckpt_callbacks, override_hidden_dims
from data.loader import DataSplit
import torch

from run.plot_curves import plot_batch, plot_mixed_context, plot_normal_with_context, plot_z_analysis

def merge_context(data_split : DataSplit):
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

    new_context = Seq2SeqSamples(key=data_split.context.key, observed_start=data_split.context.observed_start, observed=new_observed, future_len=data_split.context.future_len, offset=data_split.context.offset, future=new_future)
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
        return lambda x : collate_sampled_context(x, ncontext=ncontext)

def create_my_own_waves(n: int):
    """
    Create a dataset of n waves. Each wave is a sine wave with random phase, frequency and scale.
    Args:
        n: the number of waves to create

    Returns: a numpy array of shape (20, n*2, 1) containing the waves and their clamped pairs

    """
    length = 20

    waves = np.zeros((length, n * 2, 1))
    # Create rng for the waves
    rng = np.random.default_rng(seed=1)

    for i in range(n):
        scale = 1
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 1)

        # Create the wave and sample at time points from 0 to length-1
        wave = scale * np.sin(np.arange(length) * freq + phase)

        # Create a new wave that is the same wave, but the last 5 time points are constant
        wave2 = np.copy(wave)
        wave2[-5:] = wave2[-5]

        waves[:, i * 2, 0] = wave
        waves[:, i * 2 + 1, 0] = wave2
    # Cast to float32
    waves = waves.astype(np.float32)
    return waves

def get_custom_dataset_for_validation(args, ncontext):
    """
    Returns all data needed for my custom dataset: the train and val loaders,
    the wave np arrays and train/val Dataset objects for the dataset.

    Args:
        args: the arguments, defined by the user. In particular, contains the future length,
        batch size and whether to merge context
        ncontext: the number of samples to put into the context

    Returns: a tuple with all required loaders, datasets, np arrays

    """

    new_waves = create_my_own_waves(50000)
    val_cnt = 9000
    val_waves = new_waves[:, :val_cnt, :]
    train_waves = new_waves[:, val_cnt:, :]

    train_set = SyntheticGlancingSameContext(train_waves, args.future_len, args.batch_size)
    val_set = SyntheticGlancingSameContext(val_waves, args.future_len, args.batch_size)

    val_loader = DataLoader(
        val_set, shuffle=False, batch_size=args.batch_size,
        collate_fn=(get_collate_function(args.my_merge_context, ncontext))
    )

    loader = DataLoader(
        train_set, shuffle=False, batch_size=args.batch_size,
        collate_fn=(get_collate_function(args.my_merge_context, ncontext))
    )
    return loader, val_loader, train_waves, val_waves, train_set, val_set


def train(train_set: SyntheticGlancingSameContext, outroot: Path, args: Namespace):
    """
    Train the model on the given training set.

    Args:
        train_set: the training set
        outroot: the output root directory
        args: parameters for training
    """
    # Initialize the model
    process = init_model(args, sp_cls=SPSystemBase)

    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints-synthetic"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Create dataloader and pass to trainer
    ncontext = args.batch_size // 4
    loader = DataLoader(
        train_set, shuffle=False, batch_size=args.batch_size,
        collate_fn=get_collate_function(args.my_merge_context, ncontext)
    )

    # If we want to use proper validation, we need the custom dataset
    if args.my_use_proper_validation:
        loader, val_loader, _, _, _, _ = get_custom_dataset_for_validation(args, ncontext)
    else:
        val_loader = None       # TODO: check if this is fine

    # Plot a few samples from training
    for (i, x) in enumerate(loader):
        plot_batch(x.context, x.target)
        if i == 2:
            break

    # Create trainer and fit
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=None
    )
    trainer.fit(process, train_dataloader=loader, val_dataloaders=[val_loader] if args.my_use_proper_validation else [loader])


def test(
        test_set: SyntheticGlancingSameContext,
        args: Namespace,
        ckpt_path: Union[str, None],
        waves: np.ndarray,
        use_GM: bool = False,
        test_with_mixed_context: bool = False,
        plot_nice_mixed_context_summary: bool = False
    ):
    """
    Test the model on the given test set.

    Args:
        test_set: a dataset to test the model on
        args: arguments for the model
        ckpt_path: path to the checkpoint
        waves: the initial np array that the dataset was created from
        use_GM: whether Gaussian Mixtures were used during training / should be used now for testing
        test_with_mixed_context: whether to test the model by giving it mixed context
        plot_nice_mixed_context_summary: whether to plot a nice summary of the provided mixed context
    """

    # Initialize model
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase)
    process.freeze()

    # Create a loader for the data
    ncontext = args.batch_size // 4
    loader = DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size,
        collate_fn=get_collate_function(args.my_merge_context, ncontext)
    )

    # If we want to use proper validation, we need the custom dataset.
    if args.my_use_proper_validation:
        loader, val_loader, _, _, _, _ = get_custom_dataset_for_validation(args, ncontext)

    # If we want to test on the validation set, then change the loader to the validation loader
    if args.my_plot_validation_set:
        loader = val_loader

    # For plotting nice mixed context summary
    saved_contexts = []
    saved_targets = []
    saved_predicted_target_futures = []
    saved_predicted_target_futures_std = []
    saved_qs_context = []
    saved_target_complement_curves = []
    saved_clamped_cnt = []


    # Go over the meta samples
    for (i, meta_sample) in enumerate(loader):
        # Calculate the predictions for the given meta sample
        results = process(meta_sample)

        # Extract the future predictions of the target
        if use_GM:
            # If a GM is used we will get 2 Gaussian's as predictions
            target_future_prediction_1_mean = results.stochastic[0].loc[0, :, :, 0, 0].detach().numpy()
            target_future_predictions_1_std = results.stochastic[0].scale[0, :, :, 0, 0].detach().numpy()
            target_future_prediction_2_mean = results.stochastic[1].loc[0, :, :, 0, 0].detach().numpy()
            target_future_predictions_2_std = results.stochastic[1].scale[0, :, :, 0, 0].detach().numpy()
            k = results.stochastic[2].detach().numpy()

        else:
            # By default, we will get a single Gaussian as prediction
            target_future_prediction_mean = results.stochastic.mean[0, :, :, 0, 0].detach().numpy()
            target_future_predictions_std = results.stochastic.scale[0, :, :, 0, 0].detach().numpy()

        # Get the real underlying curves of the target (this is a hack. I.e., I do this only because I
        # know how SyntheticGlancingSameContext is implemented and since shuffle=False in the loader).
        # In particular, get the complement curves of the ground truth curves
        all_indices = [test_set.get_index(j) for j in range(i * args.batch_size, (i + 1) * args.batch_size)]
        complement_indices = [(i // 2 * 2) + (1 - (i % 2)) for i in all_indices]
        target_curves = waves[:, all_indices, 0]
        target_complement_curves = waves[:, complement_indices, 0]

        # Calculate the number of clamped curves in the context (we divide by 4, assuming the
        # context is the first 1/4 of the batch, i.e., this is hacky)
        clamped_cnt = len([i for i in all_indices[:len(all_indices) // 4] if i % 2 == 0])

        # Calculate the average of the 2 futures for every target
        target_averaged_futures = (target_curves + target_complement_curves) / 2.0
        target_averaged_futures = target_averaged_futures[-10:, :]

        # If we do not want to plot the averaged curves, set the variable to None
        if args.my_dont_plot_reds:
            target_averaged_futures = None

        # In case we want to plot a nice summary of the mixed context, save this meta-sample
        if plot_nice_mixed_context_summary:
            saved_contexts.append(meta_sample.context)
            saved_targets.append(meta_sample.target)
            saved_predicted_target_futures.append(target_future_prediction_mean)
            saved_predicted_target_futures_std.append(target_future_predictions_std)
            saved_qs_context.append(results.posteriors.q_context)
            saved_target_complement_curves.append(target_complement_curves)
            saved_clamped_cnt.append(clamped_cnt)

        # Else, immediately plot this batch
        elif not use_GM:
            if test_with_mixed_context:
                # If we are using mixed context, plot the q(z | C) additionally
                context_q = results.posteriors.q_context
                plot_batch(meta_sample.context, meta_sample.target, target_future_prediction_mean, target_future_predictions_std, target_averaged_futures, q_context=context_q, target_complement_curves=target_complement_curves, clamped_cnt=clamped_cnt)
            else:
                # If we are not using mixed context, not using GM, etc (i.e., this is the most basic case), just plot the predictions
                plot_batch(meta_sample.context, meta_sample.target, target_future_prediction_mean, target_future_predictions_std, target_averaged_futures)
        else:
            # If we are using GM, plot the predictions for both Gaussians
            plot_batch(meta_sample.context, meta_sample.target, target_future_prediction_1_mean, target_future_predictions_1_std, target_averaged_futures, target_future_prediction_2_mean, target_future_predictions_2_std)

        # We only plot 10 meta-samples. In case we care about the mixed context summary, for safety we caclulate
        # predictions for 50 meta-samples, to make sure all clamped counts are met at least twice
        if (i > 10 and not plot_nice_mixed_context_summary) or (i > 50 and plot_nice_mixed_context_summary):
            break

    # If we wanted to plot a nice summary of the mixed context, plot it now, after all 50 meta-samples have been processed
    if plot_nice_mixed_context_summary:
        plot_mixed_context(saved_contexts, saved_targets, saved_predicted_target_futures, saved_predicted_target_futures_std, saved_qs_context, saved_target_complement_curves, saved_clamped_cnt)

def test_q_of_z(
        test_set: SyntheticGlancingSameContext,
        args: Namespace,
        ckpt_path: str
):
    """
    Test the model on the given test set and plot the q(z | C) for each meta sample.

    Args:
        test_set: the dataset on which to test
        args: model arguments
        ckpt_path: path to the model to load
    """
    # Initialize model
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase)
    process.freeze()

    # Create a loader for the data
    ncontext = args.batch_size // 4
    loader = DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size,
        collate_fn=get_collate_function(args.my_merge_context, ncontext)
    )

    # If we want to use proper validation, we need the custom dataset.
    if args.my_use_proper_validation:
        loader, val_loader, _, _, _, _ = get_custom_dataset_for_validation(args, ncontext)

    # If we want to plot the validation set, change the loader to validation loader
    if args.my_plot_validation_set:
        loader = val_loader

    # Go over the meta samples
    for (i, meta_sample) in enumerate(loader):

        # Calculate the predictions for the given meta sample
        results = process(meta_sample)

        # Extract the z distribution of the context and plot it
        context_q = results.posteriors.q_context
        plot_normal_with_context(meta_sample.context, context_q)

        if i > 18:
            break

def test_z_analysis(
    test_set: SyntheticGlancingSameContext,
        args: Namespace,
        ckpt_path,
        waves: np.ndarray
):
    """
    Test the model on the given test set and plot the z analysis for the given meta samples.
    Args:
        test_set: the dataset on which to test
        args: model arguments
        ckpt_path: path to the model to load
        waves: the initial np array that the dataset was created from

    """

    # Initialize model
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase)
    process.freeze()

    # Create a loader for the data
    ncontext = args.batch_size // 4
    loader = DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size,
        collate_fn=get_collate_function(args.my_merge_context, ncontext)
    )

    # If we want to use proper validation, we need the custom dataset.
    if args.my_use_proper_validation:
        loader, val_loader, train_waves, val_waves, train_set, val_set = get_custom_dataset_for_validation(args, ncontext)
        waves = train_waves
        test_set = train_set

    # If we want to plot the validation set, change the loader and waves to the validation ones
    if args.my_plot_validation_set:
        loader = val_loader
        waves = val_waves
        test_set = val_set

    # Set the number of samples to plot and the number of z samples
    meta_sample_cnt = 6
    z_sample_count = 11

    # Get the interval from which to sample z
    z_l, z_r = args.my_z_anal_left, args.my_z_anal_right

    # Sample with equal distances from interval [z_l, z_r] z_samples times
    z_samples = [z_l + (z_r - z_l) * i / (z_sample_count - 1) for i in range(z_sample_count)]

    # Create 2D python arrays of sizes (z_sample_count, meta_sample_cnt) to store the calcualted values
    targets_observed = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]  # Go over the meta samples
    target_future_predictions_mean = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]
    target_future_predictions_std = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]
    target_future_curves = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]
    target_future_complement_curves = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]

    # Go over the meta samples and the z samples
    for (i, meta_sample) in enumerate(loader):
        for j, z in enumerate(z_samples):

            # convert z to tensor and reshape it
            z = torch.tensor(z, dtype=torch.float32).view(1, 1, 1)

            # Initialize the model, denoting the forced z value
            process = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=z)

            # Calculate the predictions for the given meta sample
            results = process(meta_sample)

            # Get the observed and future parts of the target
            target_observed = meta_sample.target.observed[:, 1, 0, 0].detach().numpy() # We could do [:, xx, 0, 0], since each xx corresponds to a different curve in the target
            target_future = meta_sample.target.future[:, 1, 0, 0].detach().numpy()

            # Extract the future predictions of the target
            target_future_prediction_mean = results.stochastic.mean[0, :, 1, 0, 0].detach().numpy()
            target_future_prediction_std = results.stochastic.scale[0, :, 1, 0, 0].detach().numpy()

            # Get the 2 possible futures
            all_indices = [test_set.get_index(i) for i in range(i * args.batch_size, (i + 1) * args.batch_size)]
            complement_indices = [(i // 2 * 2) + (1 - (i % 2)) for i in all_indices]
            target_future_curve = waves[:, all_indices, 0]
            target_future_complement_curve = waves[:, complement_indices, 0]

            # Take only the last 10 points, since we care only about the future
            target_future_curve = target_future_curve[-10:, 1]
            target_future_complement_curve = target_future_complement_curve[-10:, 1]

            # Store the values for later
            targets_observed[i][j] = target_observed
            target_future_predictions_mean[i][j] = target_future_prediction_mean
            target_future_predictions_std[i][j] = target_future_prediction_std
            target_future_curves[i][j] = target_future_curve
            target_future_complement_curves[i][j] = target_future_complement_curve

        # Stop if we have enough samples
        if i == meta_sample_cnt - 1:
            break

    plot_z_analysis(targets_observed, target_future_predictions_mean, target_future_predictions_std, target_future_curves, target_future_complement_curves, z_samples)

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

    # Added for this experiment
    parser.add_argument("--my_plot_q_of_z", default=False, action="store_true", help="Plot the q(z | C)")
    parser.add_argument("--my_fix_variance", type=bool, default=False, help="Checkpoints for my purposes...")
    parser.add_argument("--my_merge_context", default=False, action="store_true", help="Whether to have the context as fully observed, with empty future")
    parser.add_argument("--my_force_posteriors", default=False, action="store_true",help="Whether to force the posteriors to be different for different targets")
    parser.add_argument("--my_ckpt", type=str, help="Checkpoints for my purposes...")
    parser.add_argument("--my_merge_observed_with_future", default=False, action="store_true", help="Whether to merge observed with futures for z encoding")
    parser.add_argument("--my_plot_z_analysis", default=False, action="store_true", help="Whether to merge observed with futures for z encoding")
    parser.add_argument("--my_z_anal_left", type=float, default=-0.5, help="left interval start of z sampling")
    parser.add_argument("--my_z_anal_right", type=float, default=0.4, help="right interval start of z sampling")
    parser.add_argument("--my_use_proper_validation", default=False, action="store_true", help="Whether to use my own waves waves and have a proper validation set")
    parser.add_argument("--my_plot_validation_set", default=False, action="store_true", help="Whether to plot validation set in the test methods, instead of training set")
    parser.add_argument("--my_dont_plot_reds", default=False, action="store_true", help="Whether to not plot the expected averaging guess")
    parser.add_argument("--my_use_GM", default=False, action="store_true", help="Whether to use gaussian mixtures for the prediction")
    parser.add_argument("--my_test_with_mixed_context", default=False, action="store_true", help="Whether to use mixed context for testing")
    parser.add_argument("--my_plot_nice_mixed_context_summary", default=False, action="store_true", help="Whether to use mixed context for testing")
    parser.add_argument("--my_train_with_mixed_context", default=False, action="store_true", help="Whether to use mixed context for testing")

    # Add Trainer args
    parser = Trainer.add_argparse_args(parser)

    # Add model specific args
    parser = SPSystemBase.add_model_specific_args(parser)
    args = parser.parse_args()

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
    waves_path = dataset_dir / args.waves_file

    # Load the raw sinusoid waves
    waves = np.load(waves_path)

    # Train on phases from [0, 2*pi)
    logging.info("Training")
    train_set = SyntheticGlancingSameContext(waves, args.future_len, args.batch_size, args.my_test_with_mixed_context or args.my_train_with_mixed_context)

    # Update args with values needed for loading models
    args.enc_nhid = 5
    args.no_pool = True
    args.observed_len = waves.shape[0] - args.future_len
    args.nposes = 1
    args.fix_variance = args.my_fix_variance

    # Train
    if not args.test:
        train(train_set, outroot, args)

    # Evaluate
    if args.test and args.my_plot_q_of_z:
        test_q_of_z(train_set, args, args.my_ckpt)
    elif args.test and args.my_plot_z_analysis:
        test_z_analysis(train_set, args, args.my_ckpt, waves)
    elif args.test:
        test(train_set, args, args.my_ckpt, waves, args.my_use_GM, args.my_test_with_mixed_context, args.my_plot_nice_mixed_context_summary)


if __name__ == "__main__":
    main()

r"""

Not fixed variance:
Results:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.25 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 2 --log_file final_train_log-5.txt --test --my_fix_variance False --my_ckpt "saved-experiments/synthetic-glancing-same-context/2-not-fixed-variance/checkpoint.ckpt"
Loss curves:
python3 .\run\plot_curves.py "saved-experiments/synthetic-glancing-same-context/2-not-fixed-variance/train_log.txt" train
python3 .\run\plot_curves.py "saved-experiments/synthetic-glancing-same-context/2-not-fixed-variance/train_log.txt" valid

Fixed variance:
Results:
python3 .\run\plot_curves.py "saved-experiments\synthetic-glancing-same-context\1-fixed-variance\train_log.txt" train
python3 .\run\plot_curves.py "saved-experiments\synthetic-glancing-same-context\1-fixed-variance\train_log.txt" valid
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.25 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 2 --log_file final_train_log-3.txt --test --my_fix_variance True --my_ckpt "saved-experiments/synthetic-glancing-same-context/1-fixed-variance/checkpoint.ckpt"

Z posterior: (not fixed variance)
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.25 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 2 --log_file final_train_log-5.txt --test --my_fix_variance False --my_ckpt "saved-experiments/synthetic-glancing-same-context/2-not-fixed-variance/checkpoint.ckpt" --my_plot_q_of_z --test

Z posterior (fixed variance)
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.25 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 2 --log_file final_train_log-3.txt --test --my_fix_variance True --my_ckpt "saved-experiments/synthetic-glancing-same-context/1-fixed-variance/checkpoint.ckpt" --my_plot_q_of_z --test


What I think happens:
    1. Model minimizes the KL between (q(z | context) || q (z | target)) by making the posterior of the context and target to some constant value.
    That can be seen from the very beginning, since the KL is pretty much always is 0. That comes from the fact, that in the z_encoder, "shared_layers" contains
    ReLU activation, and from the beginning, it does make shared_rep=0, which results in the posterior to be a constant distribution both for the context and target encodings.
    2. Therefore, the process decoder essentially loses the information from z, which in turn results in it having to learn averaged predictions. 
    
Solutions:
    1. Increase the weight on KL, so that the model is forced to make the posterior of the context and target to be different. Although this may result in it always making the 
    shared_rep (or others) to be 0, which in turn will result in the same problem. Let's try this.
 python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 2 --log_file final_train_log-MLP-1.txt --my_fix_variance False --batch_size 12 --my_plot_q_of_z --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=5-v1.ckpt"
    2.
Had to change z_n_hid
Had to merge the context
 python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=083-v2.ckpt"
    
    3. Did not work as well. So NOW, I will try to force posteriors myself, to see if it can learn from them. I.e., I will set the posterior to N(0, 0.001) if context is 
    a normal sin continuation and set the posterior to N(1, 0.001) if it is not.

TRY: change the context encoding to 0/1 (but not the target encoding! i.e., we allow for learning the target encoding. So we fix only one part of the KL, and learn the other)
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_force_posteriors

FOUND ISSUE: the context z encoding and the target z encoding both encode ONLY the observed parts of the sequences. In this situation, that does not make sense, since the observed part does not store the required information.
SOLUTION: encode the context/target into p(z) by encoding on merged observed+target.
ISSUE: this gives the future information for z, which can then be used by the decoder. BUUUUT, since z is one-dimensional here, this might be fine
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_force_posteriors --my_merge_observed_with_future


FINALLY, SOMETHING THAT IS WORKING!!!
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_force_posteriors --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=042-v5.ckpt"
Note that here the posteriors are properly learned. Adding a validation set could be tried, but at least it does not underfit!

Now, try to do the same thing just without posterior forcing!
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future 
AND BAAAAAM: p(z | C is capped) = N(-0.45, (0.1) AND p(z | C continues) = N(0.45, 0.1) this is after epoch 3
After epoch 8, its std=0.1, Mean is -1.1 or 0.0. 
Pretty much exactly like this!!!!!!!
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=016-v7.ckpt" 
Now it's time to try search for the boundary of z.


NOW THIS IS ON TRAINING DATA VARYING Z: (this is exactly what we want)
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-MLP-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=016-v7.ckpt" --my_plot_z_analysis 
=========== MAIN RESULT ^^^^ ==========




Aftwerwards, now we can include a validation set! and run all of this with a proper validation
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file FFINAAALL-with-proper-val-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_plot_z_analysis --my_use_proper_validation
(ran for full 40 epochs)
BTW INDEED, setting hid dim to low-enough values makes KL be 0.0! Most probably due to ReLU.
so the best model is at "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=025-v6.ckpt", so to test it:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file FFINAAALL-with-proper-val-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=025-v6.ckpt" --my_use_proper_validation --my_dont_plot_reds 
This one is doing very poorly for the straight lines. But in principle, it's fine..
Now do the same thing, just this time, plot what was the case with the validation set:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file FFINAAALL-with-proper-val-1.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=025-v6.ckpt" --my_use_proper_validation --my_plot_validation_set --my_dont_plot_reds 

results:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.1 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file FFINAAALL-with-proper-val-6.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_use_proper_validation
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file FFINAAALL-with-proper-val-6.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=014-v9.ckpt" --test --my_use_proper_validation --my_plot_validation_set --my_dont_plot_reds 
Results kinda suck, probably need more data.

Let's train with more data...

python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file FFINAAALL-with-proper-val-7.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --test --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=003-v13.ckpt" --test --my_use_proper_validation --my_dont_plot_reds 

The most 


realistic demonstration:
python -m run.run_synthetic_glancing_same_context  --component MLP --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file FFINAAALL-with-proper-val-9.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_use_proper_validation --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=007-v12.ckpt" --test --my_dont_plot_reds --my_plot_z_analysis --my_z_anal_right 1.57 --my_z_anal_left 0.36 [--my_plot_validation_set]


Without fixing variance, it still works the same:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 32 --log_file final_train_log-RNN-69txt --my_fix_variance False --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=4-monitored_nll=0.384.ckpt" --test --my_dont_plot_reds --my_plot_z_analysis

Outputting a GM:
C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=005-v14.ckpt
 python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file FFINAAALL-with-proper-val-9.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_use_proper_validation --my_ckpt "C:\Users\augus\Desktop\social-processes-experiments\out-2\logs\checkpoints-synthetic\last-epoch=005-v14.ckpt" --test --my_use_GM --my_dont_plot_reds
 
outputting gm:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 6 --log_file FFINAAALL-with-proper-val-9.txt --my_fix_variance True --batch_size 12 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_use_GM --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=4-monitored_nll=-1.100.ckpt" --test --my_dont_plot_reds
 
 
 
4 things in the context:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-69txt --my_fix_variance True --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future
testing:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --my_fix_variance True --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=10-monitored_nll=-1.382.ckpt" --test --my_test_with_mixed_context --my_dont_plot_reds
z analysis of the corresponding points:
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --my_fix_variance True --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=10-monitored_nll=-1.382.ckpt" --test --my_plot_z_analysis --my_z_anal_left 0.21 --my_z_anal_right 1.75

Better z analysis for testing with mixed context (train with same context):
python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --my_fix_variance True --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=10-monitored_nll=-1.382.ckpt" --test --my_test_with_mixed_context --my_dont_plot_reds --my_plot_nice_mixed_context_summary

When trained with mixed context:
 python -m run.run_synthetic_glancing_same_context --gpus 1 --future_len 10 --waves_file phased-sin-with-stops.npy --outdir out-2 --skip_normalize_rot --data_dim 1 --dropout 0.0 --max_epochs 250 --r_dim 2 --z_dim 1 --override_hid_dims --hid_dim 5 --log_file final_train_log-RNN-691.txt --my_fix_variance True --batch_size 16 --skip_deterministic_decoding --my_merge_context --my_merge_observed_with_future --my_ckpt "out-2\logs\checkpoints-synthetic\mon-epoch=13-monitored_nll=23.385.ckpt" --my_train_with_mixed_context --test --my_test_with_mixed_context --my_dont_plot_reds --my_plot_nice_mixed_context_summary
 
 
out-2\logs\checkpoints-synthetic\mon-epoch=13-monitored_nll=23.385.ckpt

"""

# https://jmtomczak.github.io/