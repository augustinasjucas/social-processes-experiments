from data.types import ModelType, ComponentType
from common.utils import EnumAction
import numpy as np
from torch.utils.data import DataLoader
from data.datasets import SyntheticGlancingSameContext
from data.loader import collate_sampled_context
from data.types import Seq2SeqSamples
from data.loader import DataSplit
import torch


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


def add_synthetic_glancing_same_context_arguments(parser):
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

    parser.add_argument("--my_use_proper_validation", default=False, action="store_true", 
                        help="Use custom waves to create and use a proper validation set.")
    parser.add_argument("--my_merge_context", default=False, action="store_true", 
                        help="Change the dataset's context so that it contains only fully-observed sequences with empty futures.")
    parser.add_argument("--my_ckpt", type=str, help="Specify the absolute path to the checkpoint.")
    parser.add_argument("--my_merge_observed_with_future", default=False, action="store_true", 
                        help="Merge observed data with future data for z encoding.")
    parser.add_argument("--my_use_GM", default=False, action="store_true", 
                        help="Use a two Gaussian Mixture for prediction instead of a single gaussian")
    parser.add_argument("--my_train_with_mixed_context", default=False, action="store_true", 
                        help="Train using mixed context.")
    return parser

def add_synthetic_glancing_same_context_arguments_for_testing(parser):
    # General testing arguments
    parser.add_argument("--my_plot_validation_set", default=False, action="store_true", help="Plot the validation set in test methods instead of the training set.")
    parser.add_argument("--my_test_with_mixed_context", default=False, action="store_true", help="Use mixed context for testing: instead of having only clamped or "
                        + "only continued sequences in the context, have some clamped and some continued sequences.")
    parser.add_argument("--my_plot_q_of_z", default=False, action="store_true", help="Enable plotting of the approximate distribution q(z | C) for the context representation.")
    parser.add_argument("--my_plot_only_some_batch", default=False, action="store_true", help="Plot only a subset of context and target data instead of all data.")

    # Arguments for generating large-plot z-analysis figures
    parser.add_argument("--my_plot_z_analysis", default=False, action="store_true", help="Generate plots for analyzing effects of different sampled z values")
    parser.add_argument("--my_z_anal_left", type=float, default=-0.5, help="Set the left interval end for z sampling.")
    parser.add_argument("--my_z_anal_right", type=float, default=0.4, help="Set the right interval end for z sampling.")
    parser.add_argument("--my_plot_z_sample_count", type=int, default=11, help="Specify the number of z samples to use for z analysis.")

    # Additional arguments for plotting z analysis on a single, consolidated plot
    parser.add_argument("--my_plot_tight_z_analysis", default=False, action="store_true", help="Plot z analysis on a single, consolidated plot.")

    # Arguments for generating a plot which for each type of context in the dataset, shows examples of the context, the q(z|C) and the predictions
    parser.add_argument("--my_plot_nice_mixed_context_summary", default=False, action="store_true", help="Generate a summary plot for mixed context testing.")

    # Arguments for testing what effect does sampling from q(z | C) have on the output predictions
    parser.add_argument("--my_test_multiple_samples_of_z", default=False, action="store_true", help="Use multiple samples from q(z | C) for testing: plot outputs for each of the samples.")
    parser.add_argument("--my_dont_plot_reds", default=False, action="store_true", help="Disable plotting of the expected averaging guess.")
    parser.add_argument("--my_multi_plot_greens", default=False, action="store_true", help="Enable plotting of multiple green samples.")
    parser.add_argument("--my_multi_plot_reds", default=False, action="store_true", help="Enable plotting of multiple red samples.")
    parser.add_argument("--my_multi_plot_blacks", default=False, action="store_true", help="Enable plotting of multiple black samples.")
    parser.add_argument("--my_multi_plot_greens_stds", default=False, action="store_true", help="Enable plotting of standard deviations for green samples.")
    parser.add_argument("--my_multi_sample_count", type=int, default=50, help="Specify the number of samples to draw.")

    return parser