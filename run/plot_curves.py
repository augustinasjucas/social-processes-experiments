from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
from data.types import Seq2SeqSamples
import torch
import matplotlib.patheffects as pe

def plot_batch(context: Seq2SeqSamples, target:Seq2SeqSamples,
               predicted_target_futures: np.ndarray = None,
               predicted_target_futures_std: np.ndarray = None,
               averaged_future_target_curves: np.ndarray = None,
               predicted_target_futures_2: np.ndarray = None,
               predicted_target_futures_std_2: np.ndarray = None,
               k: np.ndarray = None,
               q_context: torch.distributions.Normal = None,
               target_complement_curves: np.ndarray = None,
               clamped_cnt: int = 0):
    """
    Used for plotting the results of a single meta-sample.

    Plots the context on the left.

    Plots the target on the right. If provided, the following can also be plotted in the same target subplot:
        - the predicted target futures (and stds)
        - the average between the 2 possible futures
        - GM part of the prediction: this would plot 2 predicted curves with different transparency values
        according to how much each of them is favoured according to the GM
        - q_context: a third column for plotting q(z | C)
        - target_complement_curves: the complement of the expected future

    Args:
        context: the context to plot
        target: the target to plot
        predicted_target_futures: the target futures to plot. I.e., the ground truth future. Shape: (target_future_len, target_cnt)
        predicted_target_futures_std: the standard deviations of the target futures to plot. Shape: (target_future_len, target_cnt)
        averaged_future_target_curves: the averaged future target curves to plot. Shape: (target_future_len, target_cnt)
        predicted_target_futures_2: the second prediction in case GM is used.
        predicted_target_futures_std_2: the standard deviations of the second prediction in case GM is used.
        k: the mixing coefficient of the GM
        q_context: the q distribution to plot
        target_complement_curves: the complement of the expected future
        clamped_cnt: the number of clamped sequences in the context
    """
    # In case GM is used, k should be a . If GM is not, used, then I just use k=1
    if k is not None:
        k = float(k)
    else:
        k = 1.0

    # How many context and target sequences to plot (I don't plot more than 10)
    context_cnt = min(context.observed.shape[1], 10)
    target_cnt = min(target.observed.shape[1], 10)

    # Create subplots with 2 columns and max(context_cnt, target_cnt) rows.
    # If q_context is not None, then create 3 columns
    col_cnt = 2 if q_context is None else 3
    fig, axs = plt.subplots(max(context_cnt, target_cnt), col_cnt, figsize=(10, 10))

    # If needed, plot the q distribution on the third column
    if q_context is not None:
        # Plot normal distribution at the top of the right column
        mean = q_context.loc.detach().numpy()[0, 0]
        std = q_context.scale.detach().numpy()[0, 0]
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        axs[0, 2].plot(x, stats.norm.pdf(x, mean, std), color="blue")

        # Set x axis to interval [-1; 1] and ticks every 0.1
        axs[0, 2].set_xticks(np.arange(0.3, 2.4, 0.1))
        axs[0, 2].set_xlim(0.3, 2.3)


    # Plot the context on the left
    for i in range(context_cnt):
        # Plot the context observed in dark red and context future in light red. Plot them on the same plot, but make sure that the future part comes just after the observed part
        axs[i, 0].plot(range(context.observed.shape[0]), context.observed[:, i, 0, 0], color="darkred")
        axs[i, 0].plot(range(context.observed.shape[0], context.observed.shape[0]+context.future.shape[0]), context.future[:, i, 0, 0], color="lightcoral")

    # Plot the target on the right
    for i in range(target_cnt):
        # Plot the target observed in dark blue and target future in light blue. Plot them on the same plot, but make sure that the future part comes just after the observed part
        axs[i, 1].plot(range(target.observed.shape[0]), target.observed[:, i, 0, 0], color="darkblue")
        axs[i, 1].plot(range(target.observed.shape[0], target.observed.shape[0]+target.future.shape[0]), target.future[:, i, 0, 0], color="lightblue")

        # If predictions are given, plot them
        if predicted_target_futures_std is not None:
            axs[i, 1].errorbar(range(target.observed.shape[0], target.observed.shape[0] + target.future.shape[0]), predicted_target_futures[:, i], predicted_target_futures_std[:, i], color="green", alpha=min(1.0,k+0.1))
        elif predicted_target_futures is not None:
            axs[i, 1].plot(range(target.observed.shape[0], target.observed.shape[0] + target.future.shape[0]), predicted_target_futures[:, i], color="green", alpha=min(k + 0.1, 1.0))

        # If the average between the possible futures is given, plot that
        if averaged_future_target_curves is not None:
            axs[i, 1].plot(range(target.observed.shape[0], target.observed.shape[0] + target.future.shape[0]), averaged_future_target_curves[:, i], color="red")

        # If the complement future is given plot it
        if target_complement_curves is not None:
            axs[i, 1].plot(range(target.observed.shape[0], target.observed.shape[0] + target.future.shape[0]), target_complement_curves[target_complement_curves.shape[0] // 2 :, i], color="lightblue")

        # Plot the GM part if needed (i.e., the second predicted curve)
        if predicted_target_futures_std_2 is not None:
            axs[i, 1].errorbar(range(target.observed.shape[0], target.observed.shape[0] + target.future.shape[0]), predicted_target_futures_2[:, i], predicted_target_futures_std_2[:, i], color="green", alpha=min(1.0, 1-k + 0.1))
        elif predicted_target_futures_2 is not None:
            axs[i, 1].plot(range(target.observed.shape[0], target.observed.shape[0] + target.future.shape[0]), predicted_target_futures_2[:, i], color="green", alpha=min(1-k + 0.1, 1))

    # Set the global legend for the color meanings
    ctx_observed = mlines.Line2D([], [], color='darkred', marker='s', ls='', label='Context observed')
    ctx_future = mlines.Line2D([], [], color='lightcoral', marker='s', ls='', label='Context future')
    trt_observed = mlines.Line2D([], [], color='darkblue', marker='s', ls='', label='Target observed')
    trg_future = mlines.Line2D([], [], color='lightblue', marker='s', ls='', label='Target future')
    trg_predicted = mlines.Line2D([], [], color='green', marker='s', ls='', label='Target predicted')
    # trg_guess = mlines.Line2D([], [], color='red', marker='s', ls='', label='Target expected averaged guess')

    # Plot the legend and plot
    plt.legend(handles=[ctx_observed, ctx_future, trt_observed, trg_future, trg_predicted], loc="upper right")
    plt.show()


def select_pairs_to_plot(contexts: List[Seq2SeqSamples], targets: List[Seq2SeqSamples],
                         predicted_target_futures: List[np.ndarray] = None,
                         predicted_target_futures_std: List[np.ndarray] = None,
                         qs_context: List[torch.distributions.Normal] = None,
                         target_complement_curves: List[np.ndarray] = None,
                         clamped_cnts: List[int] = 0):
    """
    Given a list of meta-samples (each with some possibly different number of clamped sequences in the context), this function
    selects 2 meta-samples for each number of clapmed sequences in the context. Then it retuns a flat list of these meta-samples
    for plotting.
    """

    context_map = {}

    for i in range(len(clamped_cnts)):
        if clamped_cnts[i] not in context_map:
            context_map[clamped_cnts[i]] = [[], [], [], [], [], []]
        context_map[clamped_cnts[i]][0].append(contexts[i])
        context_map[clamped_cnts[i]][1].append(targets[i])
        context_map[clamped_cnts[i]][2].append(predicted_target_futures[i])
        context_map[clamped_cnts[i]][3].append(predicted_target_futures_std[i])
        context_map[clamped_cnts[i]][4].append(qs_context[i])
        context_map[clamped_cnts[i]][5].append(target_complement_curves[i])

    # Get all clamped-counts
    context_map_keys = list(sorted(list(context_map.keys())))

    # Get the smallest list size in the map (i.e., the smallest number of representatives of some clamped count)
    min_size = min([len(context_map[k][0]) for k in context_map_keys])

    # If some clamped count does not have at least 2 meta-samples, then we cannot plot...
    if min_size < 2:
        # Print error and exit
        print("Not enough samples to plot mixed context")
        quit()

    min_size = 2

    # Create lists that contain in order, the first 2 elements of each list in the map
    contexts = [context_map[k][0][:min_size] for k in context_map_keys]
    targets = [context_map[k][1][:min_size] for k in context_map_keys]
    predicted_target_futures = [context_map[k][2][:min_size] for k in context_map_keys]
    predicted_target_futures_std = [context_map[k][3][:min_size] for k in context_map_keys]
    qs_context = [context_map[k][4][:min_size] for k in context_map_keys]
    target_complement_curves = [context_map[k][5][:min_size] for k in context_map_keys]

    # Flatmap those lists
    contexts = [item for sublist in contexts for item in sublist]
    targets = [item for sublist in targets for item in sublist]
    predicted_target_futures = [item for sublist in predicted_target_futures for item in sublist]
    predicted_target_futures_std = [item for sublist in predicted_target_futures_std for item in sublist]
    qs_context = [item for sublist in qs_context for item in sublist]
    target_complement_curves = [item for sublist in target_complement_curves for item in sublist]

    return contexts, targets, predicted_target_futures, predicted_target_futures_std, qs_context, target_complement_curves


def plot_mixed_context(contexts: List[Seq2SeqSamples], targets: List[Seq2SeqSamples],
                       predicted_target_futures: List[np.ndarray] = None,
                       predicted_target_futures_std: List[np.ndarray] = None,
                       qs_context: List[torch.distributions.Normal] = None,
                       target_complement_curves: List[np.ndarray] = None,
                       clamped_cnt: List[int] = 0):
    """
    Plots an analysis of what happens when different contexts are given to the model.
    In particular, what happens when the model is given a context with a different number of clamped sequences.

    Args:
        contexts: a list of contexts (the ith item corresponds to the ith meta-sample)
        targets: a list of targets (the ith item corresponds to the ith meta-sample)
        predicted_target_futures: a list of predicted target futures (the ith item corresponds to the ith meta-sample)
        predicted_target_futures_std: a list of predicted target futures stds (the ith item corresponds to the ith meta-sample)
        qs_context: a list of q(z | C) distributions (the ith item corresponds to the ith meta-sample)
        target_complement_curves: a list of target complement curves (the ith item corresponds to the ith meta-sample)
        clamped_cnt: a list of clamped counts (the ith item corresponds to the ith meta-sample)

    Returns:

    """
    # Select the meta-samples to plot. We expect that the returned lists are all of the same sizes, in particular: 2 * k, where
    # k is the number of different clamped counts in the clamped_cnt list
    contexts, targets, predicted_target_futures, predicted_target_futures_std, qs_context, target_complement_curves = select_pairs_to_plot(
        contexts, targets, predicted_target_futures, predicted_target_futures_std, qs_context, target_complement_curves, clamped_cnt
    )

    # Create subplots with 9 columns and as many rows as meta-samples to plot
    fig, axs = plt.subplots(len(contexts), 9, figsize=(20, 20))

    # Remove all ticks
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Iterate over the meta-samples
    for i in range(len(contexts)):

        # On the first 4 columns of each row, draw the context
        ctx_cnt = context_cnt = min(contexts[i].observed.shape[1], 4)
        for j in range(ctx_cnt):
            # Plot the context observed in dark red and context future in light red. Plot them on the same plot, but make sure that the future part comes just after the observed part
            axs[i, j].plot(range(contexts[i].observed.shape[0]), contexts[i].observed[:, j, 0, 0], color="darkred")
            axs[i, j].plot(range(contexts[i].observed.shape[0], contexts[i].observed.shape[0]+contexts[i].future.shape[0]), contexts[i].future[:, j, 0, 0], color="lightcoral")

        # On the 5th column draw the q distribution                                                                         target_complement_curves.shape[0] // 2 :
        if qs_context is not None:
            mean = qs_context[i].loc.detach().numpy()[0, 0]
            std = qs_context[i].scale.detach().numpy()[0, 0]
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            axs[i, 4].plot(x, stats.norm.pdf(x, mean, std), color="blue")
            # Set x axis to interval [0; 2]
            axs[i, 4].set_xticks(np.arange(-3, 2.1, 1))

        # On the next 4 columns of each row, draw the target and the predictions
        trg_cnt = target_cnt = min(targets[i].observed.shape[1], 4)
        for j in range(trg_cnt):
            # Plot the target observed in dark blue and target future in light blue. Plot them on the same plot, but make sure that the future part comes just after the observed part
            axs[i, j+5].plot(range(targets[i].observed.shape[0]), targets[i].observed[:, j, 0, 0], color="darkblue")
            axs[i, j+5].plot(range(targets[i].observed.shape[0], targets[i].observed.shape[0]+targets[i].future.shape[0]), targets[i].future[:, j, 0, 0], color="lightblue")

            if predicted_target_futures_std is not None:
                axs[i, j+5].errorbar(range(targets[i].observed.shape[0], targets[i].observed.shape[0] + targets[i].future.shape[0]), predicted_target_futures[i][:, j], predicted_target_futures_std[i][:, j], color="green")

            elif predicted_target_futures is not None:
                axs[i, j+5].plot(range(targets[i].observed.shape[0], targets[i].observed.shape[0] + targets[i].future.shape[0]), predicted_target_futures[i][:, j], color="green")

            if target_complement_curves is not None:
                axs[i, j+5].plot(range(targets[i].observed.shape[0], targets[i].observed.shape[0] + targets[i].future.shape[0]), target_complement_curves[i][target_complement_curves[i].shape[0] // 2:, j], color="lightblue")



    # Add names to columns
    axs[0, 0].set_title("Context #1")
    axs[0, 1].set_title("Context #2")
    axs[0, 2].set_title("Context #3")
    axs[0, 3].set_title("Context #4")
    axs[0, 4].set_title("q(z|C)")
    axs[0, 5].set_title("Target #1")
    axs[0, 6].set_title("Target #2")
    axs[0, 7].set_title("Target #3")
    axs[0, 8].set_title("Target #4")

    # Add the legend
    ctx_observed = mlines.Line2D([], [], color='darkred', marker='s', ls='', label='Context observed')
    # ctx_future = mlines.Line2D([], [], color='lightcoral', marker='s', ls='', label='Context future')
    trt_observed = mlines.Line2D([], [], color='darkblue', marker='s', ls='', label='Target observed')
    trg_future = mlines.Line2D([], [], color='lightblue', marker='s', ls='', label='Target future')
    trg_predicted = mlines.Line2D([], [], color='green', marker='s', ls='', label='Target predicted')
    plt.legend(handles=[ctx_observed, trt_observed, trg_future, trg_predicted], loc="upper right")

    # Change margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0)

    # Stitch every 2 rows together (to make it clearer)
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            ss = ax.get_subplotspec()
            row, col = ss.num1 // 9, ss.num1 % 9
            if (row % 2 == 0) and (col == 0):  # upper-half row (first subplot)
                y0_upper = ss.get_position(fig).y0
            elif (row % 2 == 1):  # lower-half row (all subplots)
                x0_low, _, width_low, height_low = ss.get_position(fig).bounds
                ax.set_position(pos=[x0_low, y0_upper - height_low * 1.05, width_low, height_low])

    # Add sample names to every row, also denoting the clamped_cnt
    for i in range(len(contexts)):
        axs[i, 0].set_ylabel("Meta sample #{}".format(i) + "\n" + "Clamped count: {}".format(4 - i // 2))
        axs[i, 0].yaxis.label.set_size(5)

    # Plot
    plt.show()


def plot_normal_with_context(context: Seq2SeqSamples,
                             distribution: torch.distributions.Normal):
    """
    Given the context sequences and q(z | C), this function plots the context on the left and the distribution on the right.

    Args:
        context: the context to plot
        distribution: the distribution to plot
    """

    # Plot a max of 10 context sequences
    context_cnt = min(context.observed.shape[1], 10)

    # Create subplots with 2 columns and max(context_cnt, target_cnt) rows
    fig, axs = plt.subplots(max(context_cnt, 1), 2, figsize=(10, 10))

    # Plot the context on the left
    for i in range(context_cnt):
        # Plot the context observed in dark red and context future in light red. Plot them on the same plot, but make sure that the future part comes just after the observed part
        axs[i, 0].plot(range(context.observed.shape[0]), context.observed[:, i, 0, 0], color="darkred")
        axs[i, 0].plot(range(context.observed.shape[0], context.observed.shape[0]+context.future.shape[0]), context.future[:, i, 0, 0], color="lightcoral")

    # Calculate the mean and variance of the distribution
    mean = distribution.loc.detach().numpy()[0, 0]
    std = distribution.scale.detach().numpy()[0, 0]

    # Print the mean and std for ease of use
    print("mean =", mean)
    print("std =", std)

    # Plot normal distribution at the top of the right column
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    axs[0, 1].plot(x, stats.norm.pdf(x, mean, std), color="blue")

    # Set the hspace and plot
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def plot_z_analysis (
        targets_observed: List[List[np.ndarray]],
        target_future_predictions_mean: List[List[np.ndarray]],
        target_future_predictions_std: List[List[np.ndarray]],
        target_future_curves: List[List[np.ndarray]],
        target_future_complement_curves: List[List[np.ndarray]],
        z_samples: List[float]
):
    """
    Plots the results of what happens when different samples are used as context encoding. It plots
    the changing z values for multiple samples.

    Args:
        targets_observed: the observed targets. targets_observed[i][j] is the observed target
        for the ith z sample and jth target sequence
        target_future_predictions_mean: the predicted future targets
        target_future_predictions_std:  the standard deviations of the predicted future targets
        target_future_curves: the ground truth values for the future targets
        target_future_complement_curves: the coresponding complement curves for the ground truth values of the
        future targets
        z_samples: a list of z values that were used
    """

    meta_sample_cnt = len(targets_observed)
    z_sample_count = len(z_samples)

    # Create subplot with sample_cnt rows and z_sample columns
    fig, axs = plt.subplots(meta_sample_cnt, z_sample_count, figsize=(20, 10))

    for i in range(meta_sample_cnt):
        for j in range(z_sample_count):
            # Plot the observed
            axs[i, j].plot(range(targets_observed[i][j].shape[0]), targets_observed[i][j], color="darkblue")

            # Plot the predicted future (with stds)
            axs[i, j].errorbar(range(targets_observed[i][j].shape[0], targets_observed[i][j].shape[0] + target_future_predictions_mean[i][j].shape[0]), target_future_predictions_mean[i][j], target_future_predictions_std[i][j], color="green")
            # axs[i, j].plot(range(targets_observed[i][j].shape[0], targets_observed[i][j].shape[0] + target_future_predictions_mean[i][j].shape[0]), target_future_predictions_mean[i][j], color="green")


            # Plot the two possible futures
            axs[i, j].plot(range(targets_observed[i][j].shape[0], targets_observed[i][j].shape[0] + target_future_predictions_mean[i][j].shape[0]), target_future_curves[i][j], color="lightblue")
            axs[i, j].plot(range(targets_observed[i][j].shape[0], targets_observed[i][j].shape[0] + target_future_predictions_mean[i][j].shape[0]), target_future_complement_curves[i][j], color="lightblue")


    # Set hspace
    plt.subplots_adjust(hspace=0.2, wspace=0.01, top=0.956, bottom=0.044, left=0.023, right=0.968)

    # Add legend
    trt_observed = mlines.Line2D([], [], color='darkblue', marker='s', ls='', label='Target observed')
    trg_future = mlines.Line2D([], [], color='lightblue', marker='s', ls='', label='Real target futures')
    trg_predicted = mlines.Line2D([], [], color='green', marker='s', ls='', label='Target predicted')
    plt.legend(handles=[trt_observed, trg_future, trg_predicted], loc="upper right")

    # Add a SINGLE title above each column denoting the z value
    for j, z in enumerate(z_samples):
        z = round(z, 2)
        axs[0, j].set_title(f"z={z}")

    # Do the same for rows: add a number to the left of each row
    for i in range(meta_sample_cnt):
        axs[i, 0].set_ylabel(f"Sample {i}")

    # Remove axis ticks from all subplots
    for ax in axs.flat:
        ax.label_outer()
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the subplots
    plt.show()

def plot_multiple_samples_of_z (
        context: Seq2SeqSamples,
        target_observed: np.ndarray,
        target_future: np.ndarray,
        target_main_future_prediction_mean: np.ndarray,
        target_main_future_prediction_std: np.ndarray,
        context_q: torch.distributions.Normal,
        future_means: List[np.ndarray],
        future_stds: List[np.ndarray],

        plot_samples,
        plot_sample_stds,
        plot_samples_summary,
        plot_main_prediction

):
    # Extract variables
    context_size = context.observed.shape[1]
    z_count = len(future_means)

    # For each timepoint in the future_means calculate the mean and std of the future predictions
    timepoint_count = future_means[0].shape[0]
    future_means_from_z = np.array([np.mean([future_means[j][i] for j in range(z_count)], axis=0) for i in range(timepoint_count)])
    future_stds_from_z = np.array([np.std([future_means[j][i] for j in range(z_count)], axis=0) for i in range(timepoint_count)])

    print("STDs from the main prediction: ", target_main_future_prediction_std)
    print("STDs from the z samples: ", future_stds_from_z)

    if plot_sample_stds:
        print("The stds for the samples: ")
        for std in future_stds:
            print("    Sample stds: ", std)

    # Create subplot with columns for: contex, q(z|C), target
    fig = plt.figure()
    gs = fig.add_gridspec(context_size, 2 + context_size)
    ax_q = fig.add_subplot(gs[0, 1])
    ax_target = fig.add_subplot(gs[0:, 2:])

    # Plot the context
    for i in range(context_size):
        # Plot the context observed in dark red and context future in light red. Plot them on the same plot, but make sure that the future part comes just after the observed part
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(range(context.observed.shape[0]), context.observed[:, i, 0, 0], color="darkred")
        ax.plot(range(context.observed.shape[0], context.observed.shape[0]+context.future.shape[0]), context.future[:, i, 0, 0], color="lightcoral")

    # Plot the q distribution
    mean = context_q.loc.detach().numpy()[0, 0]
    std = context_q.scale.detach().numpy()[0, 0]
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    ax_q.plot(x, stats.norm.pdf(x, mean, std), color="blue")
    ax_q.set_xticks(np.arange(0.0, 2.1, 0.4))

    # Plot the target
    ax_target.plot(range(target_observed.shape[0]), target_observed, color="darkblue")
    ax_target.plot(range(target_observed.shape[0], target_observed.shape[0] + target_future.shape[0]), target_future, color="blue", path_effects=[pe.Stroke(linewidth=3, foreground='blue')])

    # Plot the future predictions
    for i in range(z_count):
        # Plot this thing in a single merged cell: column 2-rest, rows 0-context_size
        print("target future shape", target_future.shape)
        if plot_samples:
            ax_target.plot(range(target_observed.shape[0], target_observed.shape[0] + target_future.shape[0]), future_means[i], color="green", alpha=0.25)
            if plot_sample_stds:
                ax_target.errorbar(range(target_observed.shape[0], target_observed.shape[0] + target_future.shape[0]), future_means[i], future_stds[i], color="green", alpha=0.25)


    # Plot the mean and std of the future predictions
    if plot_samples_summary:
        ax_target.errorbar(range(target_observed.shape[0], target_observed.shape[0] + target_future.shape[0]), future_means_from_z, future_stds_from_z, color="red", alpha=0.75,  path_effects=[pe.Stroke(linewidth=3, foreground='r')])

    # Plot the main future prediction with the predicted std
    if plot_main_prediction:
        ax_target.errorbar(range(target_observed.shape[0], target_observed.shape[0] + target_future.shape[0]), target_main_future_prediction_mean, target_main_future_prediction_std, color="black", alpha=1)


    # Plot the legend
    ctx_observed = mlines.Line2D([], [], color='darkred', marker='s', ls='', label='Context observed')
    trt_observed = mlines.Line2D([], [], color='darkblue', marker='s', ls='', label='Target observed')
    trg_future = mlines.Line2D([], [], color='blue', marker='s', ls='', label='Target future (ground truth)')
    trg_predicted = mlines.Line2D([], [], color='green', marker='s', ls='', label='Target predicted means, when sampling zs')
    trg_predicted_mean = mlines.Line2D([], [], color='red', marker='s', ls='', label='The mean of all sampled predictions')
    trg_mean_predicted = mlines.Line2D([], [], color='black', marker='s', ls='', label='The prediction given mean z')

    # Set hspace, wspace, top, left, right, bottom
    plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.950, bottom=0.05, left=0.05, right=0.95)

    # Plot the legend in the upper right
    plt.legend(handles=[ctx_observed, trt_observed, trg_future, trg_predicted, trg_predicted_mean, trg_mean_predicted], loc="lower left")




    plt.show()