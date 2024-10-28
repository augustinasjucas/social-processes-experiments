import argparse
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader
import constants.paths as paths
from common.initialization import init_torch
from common.utils import configure_logging
from data.datasets import SyntheticGlancingSameContext
from lightning.processes import SPSystemBase
from run.utils import init_model, init_ckpt_callbacks, override_hidden_dims

from run.synthetic_glancing_same_context.plot_curves import plot_batch
from run.synthetic_glancing_same_context.common_methods import add_synthetic_glancing_same_context_arguments, get_custom_dataset_for_validation, get_collate_function

def train(train_set: SyntheticGlancingSameContext, outroot: Path, args: Namespace):
    """
    Train the model on the given training set.

    Args:
        train_set: the training set
        outroot: the output root directory
        args: parameters for training
    """
    print("training..")
    # Initialize the model
    process = init_model(args, sp_cls=SPSystemBase)

    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints-synthetic"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Create default loader
    ncontext = args.batch_size // 4
    loader = DataLoader(
        train_set, shuffle=False, batch_size=args.batch_size,
        collate_fn=get_collate_function(args.my_merge_context, ncontext)
    )

    # If we want to use proper validation, we need a different dataset
    if args.my_use_proper_validation:
        loader, val_loader, _, _, _, _ = get_custom_dataset_for_validation(args, ncontext)
    else:
        val_loader = None       # TODO: check if this is fine

    # Plot a few samples from training
    for (i, x) in enumerate(loader):
        plot_batch(x.context, x.target)
        if i == 1:
            break

    # Create trainer and fit
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=None
    )
    trainer.fit(process, train_dataloader=loader, val_dataloaders=[val_loader] if args.my_use_proper_validation else [loader])

def main() -> None:
    """ Run the training part of the experiment """
    parser = argparse.ArgumentParser(add_help=False)

    # Add this experiment-specific arguments
    parser = add_synthetic_glancing_same_context_arguments(parser)

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

    logging.info("Training")
    train_set = SyntheticGlancingSameContext(waves, args.future_len, args.batch_size, mixed_context=args.my_train_with_mixed_context)

    # Update args with values needed for loading models
    args.enc_nhid = 5
    args.no_pool = True
    args.observed_len = waves.shape[0] - args.future_len
    args.nposes = 1

    # Train
    train(train_set, outroot, args)

if __name__ == "__main__":
    main()
