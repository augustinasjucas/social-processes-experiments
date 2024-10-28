from data.types import ModelType, ComponentType
from common.utils import EnumAction
import numpy as np
from torch.utils.data import DataLoader
from data.datasets import SyntheticGlancingSameContext
from data.loader import collate_sampled_context
from data.types import Seq2SeqSamples
from data.loader import DataSplit
import torch

from data.loader import collate_sampled_context, collate_context_independently
import itertools


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

def get_loaders(args, the_dataset):
    # Extract paramters to local variables
    people_count = args.my_people_count
    context_size = args.my_context_size
    target_size = args.my_target_size
    meta_sample_count_per_case = args.my_meta_sample_count_per_case
    observed_length = args.my_observed_length
    future_length = args.my_future_length
    args.observed_len = observed_length
    batch_size = context_size + target_size

    # Create the dataset (done deterministically, thus reproducible)
    print('the_dataset', the_dataset)
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

    # Split the dataset into training and validation (done deterministically, thus reproducible)
    train_size = (int(0.8 * len(full_set)) // batch_size) * batch_size
    val_size = len(full_set) - train_size
    train_dataset = torch.utils.data.Subset(full_set, range(train_size))
    val_dataset = torch.utils.data.Subset(full_set, range(train_size, train_size + val_size))

    # Create the data loaders
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

def add_ordering_arguments(parser):
    """
    Add the ordering-specific arguments to the parser.

    Args:
        parser: the parser to which the arguments should be added
        """
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
    parser.add_argument("--my_merge_context", default=False, action="store_true", help="Whether to have the context as fully observed, with empty futures")
    parser.add_argument("--my_merge_observed_with_future", default=False, action="store_true", help="Whether to merge observed with futures for z encoding")
    parser.add_argument("--my_ckpt", type=str, help="Absolute path to the checkpoint file to load")

    # Model output parameters
    parser.add_argument("--my_predict_categorical", default=False, action="store_true", help="If set, the model will predict a categorical distribution instead of a normal distribution")
    parser.add_argument("--my_use_softmax", default=False, action="store_true", help="If set, the model will use softmax as the last activation function, over the predicted mean")

    # Which dataset to use (for training/validation): only one of these flags should be set
    parser.add_argument("--my_use_ffa_dual", default=False, action="store_true", help="Use the FFA-dual dataset for training/validation")
    parser.add_argument("--my_use_dominating", default=False, action="store_true", help="Use the dominating dataset for training/validation")
    parser.add_argument("--my_use_ffa_dual_random", default=False, action="store_true", help="Use the FFA-dual-random dataset for training/validation")
    parser.add_argument("--my_use_ffa_full_random", default=False, action="store_true", help="Use the FFA-full-random dataset for training/validation")
    
    # Dataset configuration
    parser.add_argument("--my_how_many_random_permutations", type=int, default=1, help="How many underlying random sequences to generate for the dataset (possibly " + 
                        "multiple meta-samples will be created from the same sequence). This is used for the FFA-dual-random dataset, and FFA-full-random dataset." + 
                        "and not used for the FFA-dual dataset, and dominating dataset, as there, all possible underlying sequences are generated.")
    parser.add_argument("--my_meta_sample_count_per_case", type=int, default=5, help="How many meta samples to create for each underlying sequence/permutation")

    # Dataset parameters
    parser.add_argument("--my_people_count", type=int, default=5, help="How many people are interacting in a circle. Should match the data_dim parameter")
    parser.add_argument("--my_context_size", type=int, default=2, help="How many (observed, future) pairs to have in the context")
    parser.add_argument("--my_target_size", type=int, default=1, help="How many (observed, future) pairs to have in the target")
    parser.add_argument("--my_observed_length", type=int, default=2, help="Length of the observed sequence of every sample")
    parser.add_argument("--my_future_length", type=int, default=2, help="Length of the future sequence of every sample")
    parser.add_argument("--my_repeat_count", type=int, default=1, help="Amount of timesteps every speaker talks for")


    # For simple posterior collapse analysis, we can draw multiple times from q(z|C) and plot the outputs:
    parser.add_argument("--my_how_many_z_samples_to_plot", type=int, default=1, help="How many z samples to take and plot outputs for. Then in the target figure, " +
                        "every 2 columns will correspond to a different z sample.")
    parser.add_argument("--my_draw_out_of_distribution_z", default=False, action="store_true", help="Whether to draw a completely random z (not from q(z|C), but from N(0, 1). " +
                        "This sample would correspond to the last column in the target figure.")

    parser.add_argument("--my_test_on_validation", default=False, action="store_true")
    parser.add_argument("--my_context_independent_from_target", default=False, action="store_true")


    # Flags to generate figures for analyzing how a model performs on different datasets
    parser.add_argument("--my_analyse_datasets", default=False, action="store_true",  help="Whether to generate a figure depicting losses for a model tested on different datasets.")
    parser.add_argument("--my_plot_posteriors", default=False, action="store_true", help="Whether to generate a figure of a bunch of q(z|C) distributions for every dataset")
    parser.add_argument("--my_ffa_dual_model", type=str, default="", help="Path to the model trained on the dual dataset, used for dataset analysis")
    parser.add_argument("--my_ffa_dual_random_model", type=str, default="", help="Path to the model trained on the dual_random dataset, used for dataset analysis")
    parser.add_argument("--my_ffa_full_random_model", type=str, default="", help="Path to the model trained on the full_random model, used for dataset analysis")
    
    return parser