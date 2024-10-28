import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import Union

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from common.initialization import init_torch
from common.utils import configure_logging
from data.datasets import SyntheticGlancingSameContext
from lightning.processes import SPSystemBase
from run.utils import init_model, override_hidden_dims
import torch

from run.synthetic_glancing_same_context.common_methods import (add_synthetic_glancing_same_context_arguments, 
                                                                add_synthetic_glancing_same_context_arguments_for_testing)

from run.synthetic_glancing_same_context.plot_curves import (plot_batch, plot_mixed_context, plot_normal_with_context, plot_z_analysis, 
                                                             plot_multiple_samples_of_z, plot_z_analysis_on_a_single)

from run.synthetic_glancing_same_context.common_methods import get_custom_dataset_for_validation, get_collate_function


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
                context_q = results.posteriors.q_context
                plot_batch(meta_sample.context, meta_sample.target, target_future_prediction_mean, target_future_predictions_std, target_averaged_futures, plot_some=args.my_plot_only_some_batch, q_context=context_q)
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
        waves: np.ndarray,
        tight: bool = False
):
    """
    Test the model on the given test set and plot the z analysis for the given meta samples.
    Args:
        test_set: the dataset on which to test
        args: model arguments
        ckpt_path: path to the model to load
        waves: the initial np array that the dataset was created from
        tight: whether to plot analysis on a single ax

    """

    # Initialize model
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=None)
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
    meta_sample_cnt = 2
    z_sample_count = args.my_plot_z_sample_count

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
            targets_observed[i % meta_sample_cnt][j] = target_observed
            target_future_predictions_mean[i % meta_sample_cnt][j] = target_future_prediction_mean
            target_future_predictions_std[i % meta_sample_cnt][j] = target_future_prediction_std
            target_future_curves[i % meta_sample_cnt][j] = target_future_curve
            target_future_complement_curves[i % meta_sample_cnt][j] = target_future_complement_curve

        # Stop if we have enough samples
        if (i + 1) % meta_sample_cnt == 0:
            f_plot = plot_z_analysis_on_a_single if tight else plot_z_analysis
            f_plot(targets_observed, target_future_predictions_mean, target_future_predictions_std,
                   target_future_curves, target_future_complement_curves, z_samples)

            # Clear the lists and continue
            targets_observed = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]
            target_future_predictions_mean = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]
            target_future_predictions_std = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]
            target_future_curves = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]
            target_future_complement_curves = [[np.array([]) for _ in range(z_sample_count)] for _ in range(meta_sample_cnt)]



def test_multiple_samples_of_z(
        test_set: SyntheticGlancingSameContext,
        args: Namespace,
        ckpt_path,
        waves: np.ndarray,
):
    """
    Given a meta-sample, calculate q(z). Then, sample z multiple times and calculate the predictions for each z.
    Plot all of these curves. Check the standard deviation of these curves, comparing to the predicted std.

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
    z_sample_count = args.my_multi_sample_count

    # Go over the meta samples and the z samples
    for (i, meta_sample) in enumerate(loader):

        # Initialize the lists for the values
        future_means = []
        future_stds = []

        # Extract target observed and future
        target_observed = meta_sample.target.observed[:, 0, 0, 0].detach().numpy()
        target_future = meta_sample.target.future[:, 0, 0, 0].detach().numpy()

        # Process the meta sample to get q(z | C)
        process = init_model(args, ckpt_path, sp_cls=SPSystemBase)
        process.freeze()
        results = process(meta_sample)
        context_q = results.posteriors.q_context

        # Get the single predicted std by using the mean z
        mean_z = torch.tensor(context_q.loc, dtype=torch.float32).view(1, 1, 1)
        process = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=mean_z)
        process.freeze()
        results = process(meta_sample)
        target_main_future_prediction_mean = results.stochastic.mean[0, :, 0, 0, 0].detach().numpy()
        target_main_future_prediction_std = results.stochastic.scale[0, :, 0, 0, 0].detach().numpy()

        # Sample from that distribution multiple times
        for j in range(z_sample_count):

            # Sample from q(z | C)
            z = context_q.rsample()

            # convert z to tensor and reshape it
            z = torch.tensor(z, dtype=torch.float32).view(1, 1, 1)

            # Initialize the model, denoting the forced z value
            process = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=z)
            process.freeze()

            # Calculate the predictions for the given meta sample
            results = process(meta_sample)

            # Extract the future predictions of the target
            target_future_prediction_mean = results.stochastic.mean[0, :, 0, 0, 0].detach().numpy()
            target_future_prediction_std = results.stochastic.scale[0, :, 0, 0, 0].detach().numpy()

            # Save the values for plotting
            future_means.append(target_future_prediction_mean)
            future_stds.append(target_future_prediction_std)

        # Plot the results
        plot_multiple_samples_of_z(
            meta_sample.context,
            target_observed, target_future,
            target_main_future_prediction_mean, target_main_future_prediction_std,
            context_q,
            future_means, future_stds,
            args.my_multi_plot_greens, args.my_multi_plot_greens_stds, args.my_multi_plot_reds, args.my_multi_plot_blacks
        )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)

    # Add the common and the testing arguments
    add_synthetic_glancing_same_context_arguments(parser)
    add_synthetic_glancing_same_context_arguments_for_testing(parser)

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
    artefacts_dir = (Path(__file__).resolve().parent.parent.parent / "artefacts/")
    dataset_dir = artefacts_dir / "datasets/synthetic/sine-waves"
    waves_path = dataset_dir / args.waves_file

    # Load the raw sinusoid waves
    waves = np.load(waves_path)

    logging.info("Testing")
    train_set = SyntheticGlancingSameContext(waves, args.future_len, args.batch_size, args.my_test_with_mixed_context or args.my_train_with_mixed_context)

    # Update args with values needed for loading models
    args.enc_nhid = 5
    args.no_pool = True
    args.observed_len = waves.shape[0] - args.future_len
    args.nposes = 1

    # Evaluate
    if args.my_plot_q_of_z:
        test_q_of_z(train_set, args, args.my_ckpt)
    elif args.my_plot_z_analysis or args.my_plot_tight_z_analysis:
        test_z_analysis(train_set, args, args.my_ckpt, waves, tight=args.my_plot_tight_z_analysis)
    elif args.test and args.my_test_multiple_samples_of_z:
        test_multiple_samples_of_z(train_set, args, args.my_ckpt, waves)
    elif args.test:
        test(train_set, args, args.my_ckpt, waves, args.my_use_GM, args.my_test_with_mixed_context, args.my_plot_nice_mixed_context_summary)


if __name__ == "__main__":
    main()
