from argparse import Namespace
from pathlib import Path
from lightning.processes import SPSystemBase
from run.ordering.plot_orders import plot_meta_sample_multitime, plot_losses_and_posteriors
from run.utils import init_model, override_hidden_dims
import torch
from train.loss import SocialProcessLossCategorical, SocialProcessLoss

import argparse
import logging
from pytorch_lightning import Trainer, seed_everything
from common.initialization import init_torch
from common.utils import configure_logging
from data.datasets import GroupOrderingDataset, DualRandomFFADataset, DualFFADataset, \
    DominatingDataset, FullRandomFFADataset

from run.ordering.common_methods import add_ordering_arguments, get_loaders


def test(args: Namespace):
    """
    Given a dataset and a model, plots meta samples and predictions. This method does not calculate metrics,
    it is used for visual inspection of the model's predictions. For metrics, use the test_datasets method.
    """
    # Extract some hyperparameters
    context_size = args.my_context_size
    target_size = args.my_target_size
    batch_size = context_size + target_size
    ckpt_path = args.my_ckpt

    # Load the model from the checkpoint (specified in the arguments)
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase)

    # Define the dataset to use (based on the arguments)
    the_dataset = GroupOrderingDataset
    if args.my_use_ffa_dual:
        the_dataset = DualFFADataset
    elif args.my_use_dominating:
        the_dataset = DominatingDataset
    elif args.my_use_ffa_dual_random:
        the_dataset = DualRandomFFADataset
    elif args.my_use_ffa_full_random:
        the_dataset = FullRandomFFADataset
    
    # Get the train or validation loader depending on the arguments
    train_loader, val_loader, _, full_set = get_loaders(args, the_dataset)
    loader = val_loader if args.my_test_on_validation else train_loader

    # Go over the meta samples
    for (i, meta_sample) in enumerate(loader):
        if i < 6:
            continue

        # Calculate the predictions for the given meta sample
        results = [process(meta_sample)]

        # In case we want to draw multiple times from q(z|C), draw the remaining samples and save them for plotting
        for i in range(args.my_how_many_z_samples_to_plot - 1):
            # Sample from q(z|C)
            z = torch.tensor(results[0].posteriors.q_context.rsample([1]))

            # Initialize the model, denoting the forced z value, and calculate the prediction
            process_new = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=z)
            results.append(process_new(meta_sample))

        # In case we want to draw a completely random z (not from q(z|C), but from N(0, 1), we do that 
        if args.my_draw_out_of_distribution_z:
            z = torch.randn_like(results[0].posteriors.q_context.rsample([1]))
            process_new = init_model(args, ckpt_path, sp_cls=SPSystemBase, forced_z=z)
            results.append(process_new(meta_sample))

        # Extract the underlying permutation of this meta sample
        permutation = full_set.get_original_permutation(i * batch_size)

        # Pass all of the data to plotting method which will plot the meta sample and the predictions
        plot_meta_sample_multitime(meta_sample, permutation, context_size, batch_size, results,
                                   final_result_is_random=args.my_draw_out_of_distribution_z,
                                   predictions_are_normal=not args.my_predict_categorical)

        if i > 50:
            break

def test_datasets(args: Namespace):
    """
    Given a model (specified in the arguments), test it on multiple datasets, plot and print the losses.
    This method also takes a bunch of samples from each dataset and extracts the q(z|C) distributions for plotting
    and then plots them.

    We expect args to have the following arguments defined:
        - my_ffa_dual_model
        - my_ffa_dual_random_model
        - my_ffa_full_random_model
    It is fine if some of these are not defined, in that case the model will not be tested on that dataset.
    """
    # Define the test datasets to test the model on
    test_datasets = [DominatingDataset]

    # Extract the trained model paths
    dual_model_path = args.my_ffa_dual_model
    dual_random_model_path = args.my_ffa_dual_random_model
    full_random_model_path = args.my_ffa_full_random_model
    trained_models = []
    if dual_model_path != "":
        trained_models.append(("Dual", dual_model_path))
    if dual_random_model_path != "":
        trained_models.append(("Dual{-}random", dual_random_model_path))
    if full_random_model_path != "":
        trained_models.append(("Full{-}random", full_random_model_path))

    # Initialize the lists for losses and posteriors (q(z|C) distributions)
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

            # Get the dataset validation loader
            _, val_loader, _, _ = get_loaders(args, dataset)

            # Evaluate the average loss for the model on the dataset
            total_loss = 0
            total_count = 0
            loss = SocialProcessLossCategorical() if args.my_predict_categorical else SocialProcessLoss()
            for (i, meta_sample) in enumerate(val_loader):
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

    # Plot the losses and the posteriors
    plot_losses_and_posteriors(losses, posteriors, trained_models, test_datasets, args)


def main() -> None:
    """ Run the main experiment """
    parser = argparse.ArgumentParser(add_help=False)

    # Add the arguments specific to this experiment
    parser = add_ordering_arguments(parser)

    # Add trainer arguments
    parser = Trainer.add_argparse_args(parser)

    # Add model specific args
    parser = SPSystemBase.add_model_specific_args(parser)
    args = parser.parse_args()

    # Update arguments
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
    logging.info("Testing")

    if args.my_analyse_datasets:
        test_datasets(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
