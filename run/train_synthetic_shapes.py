import argparse

import matplotlib
import torch.cuda
from matplotlib.patches import Ellipse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from common.initialization import init_torch
from common.model import summarize
from common.utils import EnumAction, configure_logging
from data.types import FeatureSet, ModelType, ComponentType
from lightning.builders import RecurrentBuilder
from lightning.data import ShapeWalkingDataModule
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from lightning.processes import SPSystemBase
from models.attention import AttentionType, QKRepresentation
from run.utils import add_commmon_args, init_ckpt_callbacks, init_model


def plot(data_loader, predictions=None, limit=16, plot_deviations=False, cap_targets=None):
    """Plots the dataset

    Args:
        data_loader: contains the dataset to plot
        predictions: the list of predictions
        limit (int, optional): the maximum number of meta samples to plot
        plot_deviations (bool): whether to plot the standard deviations of the predictions
        cap_targets (int, optional): the maximum number of targets to plot
    """
    # Calculate the number of rows and columns we need to display the required number of meta samples
    length = int(np.ceil(np.sqrt(min(len(data_loader), limit))))

    # Create the prediction iterator
    pred_it = iter(predictions) if predictions is not None else None

    # Create the subplots and add a title
    fig, axs = plt.subplots(length, length, figsize=(10, 10), squeeze=False)
    fig.suptitle("Metasamples from the dataset")

    # Go through the dataset and plot the walks
    for i, (split) in enumerate(data_loader):
        if i >= limit:
            break

        row = int(i / length)
        col = i % length
        # Plot the context walks
        for j in range(split.context.observed.shape[1]):
            observed = split.context.observed[:, j, 0, :]
            future = split.context.future[:, j, 0, :]
            axs[row][col].scatter(observed[:, 0], observed[:, 1], color="red", s=2)
            axs[row][col].scatter(future[:, 0], future[:, 1], color="lightcoral", s=2)

        # Plot the target walks. We cap the number of targets to plot if needed
        target_walk_cnt = min(split.target.observed.shape[1], cap_targets) if cap_targets is not None else split.target.observed.shape[1]
        for j in range(target_walk_cnt):
            observed = split.target.observed[:, j, 0, :]
            future = split.target.future[:, j, 0, :]
            axs[row][col].scatter(observed[:, 0], observed[:, 1], color="blue", s=2)
            axs[row][col].scatter(future[:, 0], future[:, 1], color="lightblue", s=2)

        if predictions is not None:
            prediction = next(pred_it)
            for j in range(target_walk_cnt):
                mean = prediction.stochastic.mean.detach().numpy()[0, :, j, 0, :]
                std = prediction.stochastic.stddev.detach().numpy()[0, :, j, 0, :]
                axs[row][col].scatter(mean[:, 0], mean[:, 1], color="green", s=2)

                # Plot the standard deviations if needed
                if not plot_deviations:
                    continue
                for m, s in zip(mean, std):
                    ellipse = Ellipse((m[0], m[1]), 2 * s[0], 2 * s[1], alpha=0.1, color="green")
                    axs[row][col].add_patch(ellipse)

        # Add title for this subplot
        axs[row][col].set_title(f"Metasample {i}")

        # Make the axes of equal tick size
        axs[row][col].set_aspect("equal")

    # Add legend
    ctx_observed = mlines.Line2D([], [], color='red', marker='s', ls='', label='Context observed')
    ctx_future = mlines.Line2D([], [], color='lightcoral', marker='s', ls='', label='Context future')
    trt_observed = mlines.Line2D([], [], color='blue', marker='s', ls='', label='Target observed')
    trg_future = mlines.Line2D([], [], color='lightblue', marker='s', ls='', label='Target future')
    trg_predicted = mlines.Line2D([], [], color='green', marker='s', ls='', label='Target predicted')
    plt.legend(handles=[ctx_observed, ctx_future, trt_observed, trg_future, trg_predicted], loc="upper right")
    plt.show()


def get_default_data_parameters():
    """Returns the parameters for the dataset"""
    parameters = argparse.Namespace()
    parameters.train_meta_sample_count = 100
    parameters.train_walk_count_per_meta_sample = 6
    parameters.train_walk_length = 16
    parameters.train_seed = 10
    parameters.train_ignore_rectangular = False

    parameters.val_meta_sample_count = 10
    parameters.val_walk_count_per_meta_sample = 6
    parameters.val_walk_length = 16
    parameters.val_seed = 11
    parameters.val_ignore_rectangular = False
    return parameters

def get_default_model_parameters():
    parser = argparse.ArgumentParser(add_help=False)
    # Set the parameters for the model
    parser = add_commmon_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = SPSystemBase.add_model_specific_args(parser)
    parser.add_argument("--feature_set", type=FeatureSet, action=EnumAction,    # Make sure this is added, since otherwise it will throw an error
                        default=None, help="set of expected features")
    args = parser.parse_args()

    # Override the parameters
    builder = RecurrentBuilder
    args.model = ModelType.SOCIAL_PROCESS
    args.component = ComponentType.RNN
    args.max_epochs = 100
    args.min_epochs = 100
    args.weights_save_path = None

    args.nz_samples = 1                                 # Number of samples to take from the latent space
    args.fix_variance = False                           # Do not fix the variance of the output distribution
    args.share_target_encoder = False
    args.skip_normalize_rot = True                      # Do not normalize the rotation. I.e., no assumptions about the input of data
    args.skip_deterministic_decoding = False
    args.nlayers = 2                                    # Number of hidden layers to use in the encoder and decoder ?????
    args.nz_layers = 2                                  # Number of hidden layers in the z encoder MLP              ??????
    args.enc_nhid = 64                                  # Number of hidden units in the encoder                     ??????
    args.dec_nhid = 65                                  # Number of hidden units in the decoder                     ??????
    args.r_dim = 64                                     # Dimension of the deterministic representation of the context
    args.z_dim = 64                                     # Dimension of the latent representation of the context
    args.data_dim = 2                                   # How many features are in a single timestep (2, since we have 2D data)
    args.use_deterministic_path = False                 # Whether to also use the deterministic encoding of the context
    args.lr = 0.001                                     # Learning rate
    args.schedule_lr = False
    args.lr_steps = None
    args.lr_gamma = 0.3
    args.weight_decay = 0.0005
    args.reg = 1e-06
    args.dropout = 0
    args.teacher_forcing = 0.5
    args.attention_type = AttentionType.UNIFORM         # Do not use Attention
    args.attention_rep = QKRepresentation.IDENTITY      # Do not use Attention
    args.attention_qk_dim = 32
    args.attention_nheads = 8
    args.ndisplay = 25                                  # Likely not used at all
    args.nposes = 1                                     # Hopefully this is not used, since no pooling was used
    args.pooler_nhid_embed = 64                         # Hopefully this is not used, since no pooling was used
    args.pooler_nhid_pool = 64                          # Hopefully this is not used, since no pooling was used
    args.pooler_nout = 64                               # Hopefully this is not used, since no pooling was used
    args.pooler_stride = 1                              # Hopefully this is not used, since no pooling was used
    args.pooler_temporal_nhid = 64                      # Hopefully this is not used, since no pooling was used
    args.no_pool = True                                 # Do not use any partner pooling
    args.skip_monitoring = True

    return args

def get_default_configuration():
    # Each meta sample has 3 context walks and 3 target walks. Each walk has 8 o    bservations and 8 future steps.
    # There are both circular and rectangular walks in the dataset.
    # There are 100 train samples and 10 validation samples.
    # This is already trained (in the checkpoint)
    data_params = get_default_data_parameters()
    model_args = get_default_model_parameters()
    checkpoint = "C:\\Users\\augus\\Desktop\\social-processes-experiments\\checkpoints\\last-epoch=099.ckpt"
    return data_params, model_args, checkpoint

def get_circular_configuration():
    # Each meta sample has 3 context walks and 3 target walks. Each walk has 8 observations and 8 future steps.
    # There are ONLY circular walks in the dataset.
    # There are 100 train samples and 10 validation samples.

    # Use defaults
    data_params = get_default_data_parameters()
    model_args = get_default_model_parameters()

    # Use much smaller representations
    model_args.enc_nhid = 6
    model_args.dec_nhid = 6
    model_args.r_dim = 6
    model_args.z_dim = 6
    model_args.use_deterministic_path = True

    # Override to use circular walks ONLY!
    data_params.train_ignore_rectangular = True
    data_params.val_ignore_rectangular = True

    model_args.min_epochs = 1000
    model_args.max_epochs = 1000
    checkpoint = "C:\\Users\\augus\\Desktop\\social-processes-experiments\\checkpoints\\last-epoch=999.ckpt"
    return data_params, model_args, checkpoint

def get_second_circular_configuration():
    # Use defaults
    data_params = get_default_data_parameters()
    model_args = get_default_model_parameters()

    # Use much smaller representations
    model_args.enc_nhid = 6
    model_args.dec_nhid = 6
    model_args.r_dim = 6
    model_args.z_dim = 6
    model_args.use_deterministic_path = True
    model_args.lr = 0.01
    model_args.nz_samples = 5

    # Override to use circular walks ONLY!
    data_params.train_ignore_rectangular = True
    data_params.val_ignore_rectangular = True
    data_params.train_walk_count_per_meta_sample = 6
    data_params.train_meta_sample_count = 1000

    model_args.min_epochs = 1000
    model_args.max_epochs = 1000
    checkpoint = "C:\\Users\\augus\\Desktop\\social-processes-experiments\\checkpoints\\last-epoch=049.ckpt"
    return data_params, model_args, checkpoint
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hardcoded parameters
    to_train = False

    init_torch(42)

    # Initialize the dataset
    data_params, model_args, checkpoint = get_second_circular_configuration()
    data_module = ShapeWalkingDataModule(data_params)
    data_module.setup("fit")

    # Plot part of the training dataset
    plot(data_module.train_dataloader())

    # Initialize logging
    configure_logging("DEBUG", "logs.log")
    model = SPSystemBase(model_args, RecurrentBuilder)

    # Move model to gpu
    # model.to(device)

    # Summarize the module and the parameters
    print("Model summary:")
    summarize(model)

    # Create the logger and the callbacks
    logger = TestTubeLogger(save_dir=str("logs2.log"))
    callbacks = init_ckpt_callbacks(model_args, str("checkpoints"))

    # Train or test the model
    if to_train:
        train(model_args, logger, callbacks, checkpoint, model, data_module)
    else:
        process = init_model(model_args, checkpoint, sp_cls=SPSystemBase)
        test(process, data_module)

def train (args, logger, callbacks, checkpoint, model, data_module):
    if checkpoint is not None:
        trainer = Trainer.from_argparse_args(
            args, logger=logger, callbacks=callbacks, gpus=1 if torch.cuda.is_available() else 0,
            resume_from_checkpoint=checkpoint
        )
    else:
        trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)

    trainer.fit(model, data_module)
def test (system, data_module):
    data = data_module.train_dataloader()
    predictions = []
    for sample in data:
        predictions.append(system(sample))
    print("predictions len is ", len(predictions))
    plot(data, predictions, plot_deviations=True, cap_targets=1, limit=4)

if __name__ == "__main__":
    main()



# MSE might be bad...
# We get mean and variance.. How???