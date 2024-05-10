from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
from data.types import Seq2SeqSamples
import torch


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
