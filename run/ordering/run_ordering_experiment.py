import argparse
import logging
from argparse import Namespace
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger
import constants.paths as paths
from common.initialization import init_torch
from common.utils import configure_logging
from data.datasets import GroupOrderingDataset, DualRandomFFADataset, DualFFADataset, \
    DominatingDataset, FullRandomFFADataset
from lightning.processes import SPSystemBase
from run.ordering.plot_orders import plot_meta_sample_multitime
from run.utils import init_model, init_ckpt_callbacks, override_hidden_dims
from run.ordering.common_methods import add_ordering_arguments, get_loaders

def train(outroot: Path, args: Namespace, dataset_class):
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

    # Get the training, validation and the created dataset
    train_loader, val_loader, _, full_set = get_loaders(args, dataset_class)

    # Plot a few samples before training
    context_size = args.my_context_size
    batch_size = args.my_context_size + args.my_target_size
    for (i, x) in enumerate(train_loader):
    
        permutation = full_set.get_original_permutation(i * batch_size)
        plot_meta_sample_multitime(x, permutation, context_size, batch_size,
                                   predictions_are_normal=not args.my_predict_categorical)
    
        if i > 1:
            break

    # Create trainer
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=None
    )

    # Initialize the model
    process = init_model(args, sp_cls=SPSystemBase)

    # Train the model
    trainer.fit(process, train_dataloader=train_loader,
                val_dataloaders=[val_loader])

def main() -> None:
    """ Run the main experiment """
    parser = argparse.ArgumentParser(add_help=False)

    # Add the arguments specific to this experiment
    parser = add_ordering_arguments(parser)

    parser = Trainer.add_argparse_args(parser)

    # Add model specific args
    parser = SPSystemBase.add_model_specific_args(parser)
    args = parser.parse_args()

    # Extract the dataset class to be used
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

    # Run training
    train(outroot, args, the_dataset)

if __name__ == "__main__":
    main()
