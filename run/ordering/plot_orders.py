import numpy as np
import matplotlib.pyplot as plt
from data.types import DataSplit
import networkx as nx
import seaborn as sns

def create_graph(n,
                 observed_permutation,
                 future_permutation,
                 predicted_permutation,
                 use_loop=False):
    """
    Creates a graph out of the ordering sequences. Denotes observed, future and predicted edges differently.
    This function ended up not being used in the final version of the code. 

    Args:
        n : people count
        ...
        use_loop (bool, optional): whether to connect the last person with the first one in the graph

    Returns:
        _type_: _description_
    """
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, size=0.01)

    # Convert to numpy arrays
    observed_permutation = np.array(observed_permutation)
    future_permutation = np.array(future_permutation)

    observed_color = 'blue'
    future_color = 'blue'

    for i in range(len(observed_permutation) - 1):
        G.add_edge(observed_permutation[i], observed_permutation[i + 1], color=observed_color)

    if use_loop:
        G.add_edge(observed_permutation[-1], observed_permutation[0], color=observed_color)

    # Add the last element of the observed permutation to the start of future permutation
    future_permutation = np.insert(future_permutation, 0, observed_permutation[-1])
    print("Future permutation: ", future_permutation)
    for i in range(len(future_permutation) - 1):
        print("adding edge betwen ", future_permutation[i], " and ", future_permutation[i + 1], " with color (0, 0, 1, 1)")
        G.add_edge(future_permutation[i], future_permutation[i + 1], color=(0, 0, 1, 1))

    if predicted_permutation == []:
        pass
    elif len(predicted_permutation.shape) == 2:
        # Add the last element of the observed_permutation permutation to the start of predicted permutation
        predicted_permutation = np.insert(predicted_permutation, 0, observed_permutation[-1])
        for i in range(len(predicted_permutation) - 1):
            G.add_edge(predicted_permutation[i], predicted_permutation[i + 1], color='green')
    else:
        # the "predicted_permutation" actually contains probability distribution over the edges
        # now draw all edges from "observed_permutation[-1]" node to all other nodes with the color of the edge
        # being the probability of that edge
        for i in range(len(predicted_permutation)):
            G.add_edge(observed_permutation[-1], i, color=(0, 1, 0, predicted_permutation[i]), weight=predicted_permutation[i])


    return G

def infer_edges(context_size, meta_sample):
    """
    Knowing that interactions can be modelled by graphs, given a meta sample, creates a set
    of edges that, given the context, can be deduced with full certainty. I.e., creates a set of 
    edges which we can expect the model to always predict correctly, as they were seen in the context.
    This function is also not really used at the final version of the code.
    Args:
        context_size: context size
        meta_sample: all meta sample data
    Returns: a set of edges
    """
    # calculate what we can infer from the context alone
    known_edges_context = set()
    for i in range(context_size):
        context_sequence_observed = np.array(meta_sample.context.observed[:, i, 0, :])
        context_sequence_observed = np.argmax(context_sequence_observed, axis=-1)
        for j in range(len(context_sequence_observed) - 1):
            if context_sequence_observed[j] != context_sequence_observed[j + 1]:
                known_edges_context.add((context_sequence_observed[j], context_sequence_observed[j + 1]))

    return known_edges_context

def infer_edges_for_target(context_size, meta_sample, target_sequence_observed):
    """
    Given a meta sample with context and also the targets, creates a set of edges for each target,
    which can be deduced with full certainty.

    Args:
        context_size: context size
        meta_sample: all meta sample data

    Returns: a set of edges

    """
    target_sequence_observed = target_sequence_observed.detach().numpy()

    # Get a set of edges that can be inferred from the context
    context_edges = infer_edges(context_size, meta_sample)

    # calculate what we can infer for this target specifically, given its observed sequence
    known_edges_target = set(context_edges)
    target_sequence_observed = np.argmax(target_sequence_observed, axis=-1)
    for j in range(len(target_sequence_observed) - 1):
        if target_sequence_observed[j] != target_sequence_observed[j + 1]:
            known_edges_target.add((target_sequence_observed[j], target_sequence_observed[j + 1]))

    return known_edges_target


def draw_heatmap(
        ax,
        observed,
        future,
        predicted = None
):
    """
    For a single sequence, it draws a heatmap of the observed, future and predicted subsequences.
    Args:
        ax: the ax on which to draw
        observed: sequence of observed talking statuses. Shape is [observed_size; n_people]. Each timestep
        is a one-hot encoded vector.
        future: sequence of the future talking statuses. Shape is [future_size; n_people]. Each timestep is a
        one-hot encoded vector.
        predicted: sequence of the predicted talking statuses. Shape is [future_size; n_people]. Each timestep is a
        vector of probabilities.
    """

    # Assert correct shape lenghts
    assert len(observed.shape) == 2
    assert len(future.shape) == 2

    # Assert same people count
    assert observed.shape[1] == future.shape[1]

    # Assert future and predicted shapes match
    if predicted is not None:
        assert len(predicted.shape) == 2
        assert future.shape[0] == predicted.shape[0]
        assert future.shape[1] == predicted.shape[1]
        predicted = predicted.detach().numpy()
        predicted = predicted.T

    n_people = observed.shape[1]
    observed_length = observed.shape[0]
    future_length = future.shape[0]

    # Copy the sequences and transpose them
    observed = observed.T
    future = future.T

    # Calculate matrix size
    height = n_people
    width = observed_length + future_length

    # Create blanket matrices
    observed_matrix = np.zeros((height, width))
    predicted_matrix = np.zeros((height, width))
    observed_matrix[:, observed_length:] = np.nan
    predicted_matrix[:, :observed_length] = np.nan

    # If we have predicted values, matshow them as well
    if predicted is not None:
        # Create predicted matrix
        if predicted is not None:
            for i in range(future_length):
                predicted_matrix[:, observed_length + i] = predicted[:, i]

        # Used to be cmap="Greens"
        col = sns.color_palette("light:#f07167", as_cmap=True)
        col = "Blues"
        sns_ax = sns.heatmap(predicted_matrix, ax=ax, cmap=col, annot=True, cbar=False, linewidths=0.5, linecolor="black", square=True, annot_kws={"size": 12})

        # Apply the custom formatting to bold needed cells:
        cells_to_bold = []
        for i in range(future_length):
            future_person = np.argmax(future[:, i])
            cells_to_bold.append((int(future_person), observed_length + i))
        def bold_cell(val, cells_to_bold, cell):
            if cell in cells_to_bold:
                return "$\\bf {" + f'{val:.2f}' + "}$"  # Bold the value at (i, j)
            return f'{val:.2f}'
        for ind, text in enumerate(sns_ax.texts):
            row = ind // future_length
            col = ind % future_length + observed_length
            text.set_text(bold_cell(predicted_matrix[row, col], cells_to_bold, (row, col)))

    # Create observed matrix
    for i in range(observed_length):
        person = np.argmax(observed[:, i])
        observed_matrix[person, i] = 1
    col = sns.color_palette("light:#823b79", as_cmap=True)
    col = "Blues"
    sns.heatmap(observed_matrix, ax=ax, cmap=col, annot=False, cbar=False, square=True, linewidths=0.5, linecolor="black")

    # Go over each future timestep and draw a rectangle around the cell
    for i in range(future_length):
        future_person = np.argmax(future[:, i])
        delta = 0.0323 / 2 * 2.5 # delta is approx lw * 2 / 100
        ax.add_patch(plt.Rectangle((observed_length + i + delta, future_person + delta), 1 - 2 * delta, 1 - 2 * delta, fill=None, edgecolor=(200/255,0/255,0/255, 0.75), lw=2.5))

    # hardcode the labels :/
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels(["p1", "p2", "p3", "p4", "p5"]) 

    # set fonts of ticks
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=15.5)

    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])


def draw_inferred_graph(
        ax,
        known_edges,
        meta_sample
):
    """
    Given a meta-sample and the set of edges extracted from the context, draws a graph with those
    edges. This graph is supposed to represent which edges the model is given access to in the context.
    Not used anywhere in the final version of the code.
    """
    ax.margins(0.2)
    ax.set_box_aspect(1)
    people_count = meta_sample.context.observed.shape[3]
    G_inferred_context = nx.DiGraph()
    for i in range(people_count):
        G_inferred_context.add_node(i, label=i)
    for edge in known_edges:
        G_inferred_context.add_edge(edge[0], edge[1], color='red')
    pos = nx.circular_layout(G_inferred_context)
    colors = nx.get_edge_attributes(G_inferred_context, 'color').values()
    nx.draw(G_inferred_context, pos, with_labels=True, node_size=60, node_color='skyblue', ax=ax,
            edge_color=colors)

def draw_q_of_z(
        ax_plot,
        ax_text,
        predicted
):
    """
    Draws the q(z | C) distribution on a specified axis.
    Not used in the final version of the code.

    """
    # Draw the q(z | C)
    z_distribution_mean = np.reshape(predicted.posteriors.q_context.mean.detach().numpy(), [-1])
    z_distribution_std = np.reshape(predicted.posteriors.q_context.stddev.detach().numpy(), [-1])

    # On ax_text, write the mean and std (rounded to 4 decimals)
    if ax_text is not None:
        ax_text.axis('off')
        print(np.round(z_distribution_mean, 4))
        ax_text.text(0.5, 0.5, "Mean: " + str(np.round(z_distribution_mean, 4)),
                     horizontalalignment='center', verticalalignment='center', fontsize=8)

    slice_str = ""

    if z_distribution_mean.shape[0] != 1:
        z_distribution_mean = z_distribution_mean[0]
        z_distribution_std = z_distribution_std[0]
        slice_str = "Slice of "

    # Plot the distribution as a linspace
    z_values = np.linspace(-3, 3, 100)
    z_pdf = np.exp(-0.5 * (z_values - z_distribution_mean) ** 2 / z_distribution_std ** 2) / np.sqrt(
        2 * np.pi * z_distribution_std ** 2)
    ax_plot.plot(z_values, z_pdf)
    ax_plot.set_title(slice_str + 'q(z | C)')

    # Add axis ticks
    ax_plot.set_xticks(np.linspace(-3, 3, 7))
    ax_plot.set_xticklabels(np.round(np.linspace(-3, 3, 7), 2))
    ax_plot.axis('on')



def plot_meta_sample_multitime(
        meta_sample,
        underlying_permutation,
        context_size,
        target_size,
        predicted=None,
        final_result_is_random=False,
        predictions_are_normal=True
):
    """
    Given a meta sample (possibly with predictions), plots it: shows the context, the target observed and the predicted target.
    This function generates a figure which is used in the paper, so it is important.
    It generates two figures: one for the context and one for the target.
    """

    how_much_context_to_show = 6
    how_much_target_to_show = 2
    context_size_capped = min(context_size, how_much_context_to_show)

    target_indices_to_display = list(range(context_size_capped - 2))
    remaining_cnt = min(target_size - len(target_indices_to_display), how_much_target_to_show)

    # add the last `remaining_cnt` target sequences from the back
    for i in range(remaining_cnt):
        target_indices_to_display.append(target_size - 1 - i)
    target_indices_to_display = list(set(target_indices_to_display))

    column_count = 2
    if predicted is not None:
        column_count += (len(predicted) - 1) * 2

    # create the figures and grids for context and target
    fig_ctx, ax_ctx = plt.subplots(3, 2, figsize=(8, 8.5), sharex=True, sharey=True)
    fig_trg, ax_trg = plt.subplots(3, column_count, figsize=(8, 8.5), sharex=True, sharey=True)

    # Draw heatmaps for the context
    for i in range(context_size_capped):
        ind = (i, 0)
        if i >= 3:
            ind = (i - 3, 1)
        draw_heatmap(ax_ctx[ind], meta_sample.context.observed[:, i, 0, :], meta_sample.context.future[:, i, 0, :])

    # # I also used to draw graphs that define the context, however due to space and usefulness limitations, I don't do that anymore
    # Draw the inferred known ordering graph for the context
    # inferred_edges = infer_edges(context_size, meta_sample)
    # draw_inferred_graph(ax_ctx[2, 1], inferred_edges, meta_sample)
    # ax[2, 1].set_title("What can be inferred from the context")

    # # Draw the inferred know orderings for each target sample
    # for i, target_index in enumerate(target_indices_to_display):
    #     inferred_edges = infer_edges_for_target(context_size, meta_sample, meta_sample.target.observed[:, target_index, 0, :])
    #     draw_inferred_graph(ax[i, -1], inferred_edges, meta_sample)
    # ax[0, -1].set_title("What can be inferred for targets", fontsize=8)

    # annotate the text "Context" at x=0.25 (in relative terms) and y=1 (in relative terms) of the whole figure
    # annotation = fig.text(0.25, 1, "Context", fontsize=12, ha='center', va='center')
    # annotation = fig.annotate(text="Context", xy=(0.25, 1), xytext=(0, 20),
    #          xycoords='axes fraction', textcoords='offset points',
    #          fontsize=27, ha='center', va='baseline', fontweight='bold')

    # # I also do not draw the q(z|C), even though that is a useful thing to do
    # Draw q of z given z
    # if predicted is not None:
    #     draw_q_of_z(ax[3, 1], None, predicted[0])

    # Draw the target ground observed and ground truths
    if predicted is None:
        predicted_for_this_person = None
        for i, target_index in enumerate(target_indices_to_display):
            ind = (i, 0)
            if i >= 3:
                i -= 4
                ind = (i, 1)

            draw_heatmap(ax_trg[ind], meta_sample.target.observed[:, target_index, 0, :], meta_sample.target.future[:, target_index, 0, :], predicted_for_this_person)
    else:
        # If we have predictions, draw them as well. Note that `predicted` is a list, where each element corresponds to
        # a different the outputs from a different z samples drawn from q(z|C)
        for j in range(len(predicted)):
            for i, target_index in enumerate(target_indices_to_display):
                if predictions_are_normal:
                    predicted_for_this_person = (predicted[j].stochastic.mean[0, :, :, 0, :])[:, target_index, :]
                else:
                    predicted_for_this_person = predicted[j].stochastic.probs[0, :, :, 0, :][:, target_index, :]

                ind = (i, j*2)
                if i >= 3:
                    i -= 3
                    ind = (i, j*2+1)

                draw_heatmap(ax_trg[ind], 
                             meta_sample.target.observed[:, target_index, 0, :],
                             meta_sample.target.future[:, target_index, 0, :],
                             predicted_for_this_person)

    # Add x and y labels to lefmost and bottommost axes
    for i in range(2):
        ax_ctx[2, i].set_xlabel("Time", fontsize=20, labelpad=8)
        ax_trg[2, i].set_xlabel("Time", fontsize=20, labelpad=8)
    for i in range(3):
        ax_ctx[i, 0].set_ylabel("Person", fontsize=20, labelpad=8)
        ax_trg[i, 0].set_ylabel("Person", fontsize=20, labelpad=8)

    # make distance between axes small
    fig_ctx.subplots_adjust(hspace=0, wspace=0.1, top=1, bottom=0)
    fig_trg.subplots_adjust(hspace=0, wspace=0.1, top=1, bottom=0)
    # fig.tight_layout()

    print("Saving context and target figures to orders_ctx.pdf and orders_trg.pdf")
    fig_ctx.savefig("orders_ctx.pdf", bbox_inches="tight")
    fig_trg.savefig("orders_trg.pdf", bbox_inches="tight")

    plt.show()

flare = sns.color_palette("flare", as_cmap=True)
crest = sns.color_palette("Blues", as_cmap=True)
gt_style = {"linestyle": "--", "marker": "o", "linewidth": 3.5, "markersize": 10,
            "color": flare(0.8)}  # crest(0.9) "#0081a7"
pred_style = {"marker": "X", "linewidth": 2.5, "markersize": 12, "color": "#f07167"}  # f07167, flare(0.4)
std_border_style = {"linewidth": 1, "color": pred_style["color"], "alpha": 1, "zorder": 0}
distr_style = {"linestyle": "-", "marker": "o", "linewidth": 0.7, "markersize": 0}


def plot_losses_and_posteriors(
        losses,
        posteriors,
        trained_models,
        test_datasets,
        args
):
    """
    Given lists with losses for different models tested on different datasets, this function plots the losses.
    It also plots the q(z|C) distributions for the first 50 meta samples of each model tested on each dataset.

    Args:
        losses :    a list of length `model_cnt`, where each element is a list of length `dataset_cnt`. So losses[i][j] 
                    is the loss of model i tested on dataset j.
        posteriors : a list of length `model_cnt`, where each element is a list of length `dataset_cnt`. So posteriors[i][j]
                    is a list of length `meta_sample_cnt`, where each element is a tuple of (mean, variance) of the Normal
                    distribution q(z|C). 
                    I.e., posteriors[model][dataset][meta_sample] is a tuple of (mean, variance) which corresponds to a 
                    posterior distribution of meta-sample `meta_sample`, when model `model` was tested on dataset `dataset`.

        trained_models : a list of tuples, where each tuple is a pair of (model_name, model_path), used for labeling the plots.
        test_datasets : a list of dataset names.
        args : the arguments passed to the script
    """
  
    # other options: some_colors = [ "#1AC8ED", "#f07167", "#585B56"]
    some_colors = [crest(0.9), crest(0.9),  crest(0.9)]
    model_cnt = len(losses)
    dataset_cnt = len(losses[0])

    # Create a 2D grid of shape (dataset_cnt, model_cnt)
    width_distr = 5.1 * dataset_cnt
    height_distr = 1.5 * model_cnt

    # Calculate the width and height of the loss figure
    width_scale_for_loss = 0.3
    scale = (width_scale_for_loss / (1-width_scale_for_loss))
    width_loss = width_distr
    height_loss = height_distr / scale

    # Create the figures for losses and q(z|C)s.
    loss_fig, loss_ax = plt.subplots(dataset_cnt, figsize=(width_loss, height_loss))
    distr_fig, distr_ax = plt.subplots(model_cnt, dataset_cnt, figsize=(width_distr, height_distr))

    # Set the title for the loss figure (hardcoded :/)
    dataset_names = ["Dominating\\ dataset", "Dual"]

    for i in range(dataset_cnt):
        if dataset_cnt == 1:
            LOSS_AX = loss_ax
        else:
            LOSS_AX = loss_ax[i]

        model_name = ["Dual", "Dual-random", "Full-random"]
        
        max_loss = 0
        
        # Set title for this plot to be the name of the dataset
        LOSS_AX.set_title("Tested on $\\bf{" + dataset_names[i] + "}$", fontsize=8/scale, pad=8/scale)
        
        for j in range(model_cnt):
            # LOSS_AX.grid(linestyle="--", alpha=0.75)

            # Plot the losses for this model
            bar_ax = sns.barplot(x=[model_name[j]], y=[losses[j][i]], ax=LOSS_AX, color=some_colors[j], width=0.75, label="Trained on " + model_name[j] + " dataset") # plot with width 0.25
            bar_ax.bar_label(bar_ax.containers[j], fontsize=7/scale, fmt='%.2f', label_type='edge', padding=3)
            max_loss = max(max_loss, losses[j][i])

            # Turn off legend
            LOSS_AX.get_legend().remove()

            # Set axes
            LOSS_AX.set_ylim(bottom=0)
            LOSS_AX.set_xlim(-0.6, model_cnt - 1 + 0.6)
            LOSS_AX.set_ylabel("Loss", fontsize=10/scale)
            LOSS_AX.set_xlabel("Model training dataset", fontsize=8/scale, labelpad=8/scale)

            # Set the xticks font to smaller
            LOSS_AX.tick_params(axis='x', labelsize=7 / scale)
            LOSS_AX.tick_params(axis='y', labelsize=7 / scale)

            # Set the yticks go over every 100 steps
            LOSS_AX.set_yticks(np.arange(0, max_loss + 1, 100))
            LOSS_AX.set_ylim(top=max_loss * 1.1)

            for spine in LOSS_AX.spines.values():
                # Divide the border size by scale
                spine.set_linewidth(0.75/scale)

            # If needed, plot the q(z|C) distributions on the other figure
            if args.my_plot_posteriors:
                if dataset_cnt == 1:
                    DISTR_AX = distr_ax[j]
                else:
                    DISTR_AX = distr_ax[j, i]

                mn = np.inf
                mx = -np.inf
                slices = ""

                # Find the best dimension to slice the q(z|C) distributions
                means = np.array([mean for mean, variance in posteriors[j][i][:50]])
                slice_dimension = 0
                if means.shape[0] == 1:
                    slice_dimension = 0
                else:
                    best_variance = 0
                    for dim in range(means.shape[2]):
                        variance = np.var(means[:, 0, dim])
                        if variance > best_variance:
                            best_variance = variance
                            slice_dimension = dim

                count = len(posteriors[j][i][:50])

                # Go over the q(z|C) distributions and plot them on the axis
                for k, (mean, variance) in enumerate(posteriors[j][i][:50]):
                    std = np.sqrt(variance)

                    if len(mean.shape) == 2 and mean.shape[1] != 1:
                        std = std[:, slice_dimension]
                        mean = mean[:, slice_dimension]
                        variance = variance[:, slice_dimension]
                        slices = r"$\bf{" + str("slices") + "}$ of "

                    z_values = np.linspace(mean-4*std, mean+4*std, 100)
                    mn = min(mn, mean-3.75*std)
                    mx = max(mx, mean+3.75*std)
                    z_pdf = np.exp(-0.5 * (z_values - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)
                    z_pdf = np.reshape(z_pdf, (100,))
                    z_values = np.reshape(z_values, (100,))

                    # DISTR_AX.plot(z_values, z_pdf, color=some_colors[j])
                    sns.lineplot(x=z_values, y=z_pdf, ax=DISTR_AX, color=some_colors[j], **distr_style)

                DISTR_AX.set_title(str(count) + " " + slices + "q(z|C) distributions with model trained on $\\bf{" + trained_models[j][0] + "\\ dataset}$", fontsize=8)
                DISTR_AX.set_xlim(mn, mx)
                DISTR_AX.set_ylim(bottom=0)
                DISTR_AX.set_xlabel("z", fontsize=8)
                DISTR_AX.set_ylabel("likelihood", fontsize=8)

                DISTR_AX.tick_params(axis='x', labelsize=7)
                DISTR_AX.tick_params(axis='y', labelsize=7)
                DISTR_AX.grid(linestyle="--", alpha=0.75)

    loss_fig.tight_layout()
    distr_fig.tight_layout()
    
    # Save the figure to a pdf
    loss_fig.savefig("ds-comparison-loss.pdf")
    distr_fig.savefig("ds-comparison-distr.pdf")
    
    print("Saved loss and distribution figures to ds-comparison-loss.pdf and ds-comparison-distr.pdf")

    plt.show()


def plot_meta_sample(
        meta_sample: DataSplit,
        permutation: np.array,
        context_size,
        target_size,
        predictions=None,
        which_z_sample=0,
        people_count=None,
        plot_order_as_labels=False,
        draw_inferred_edges=False,
        plot_z_distribution=False
):
    """
    Plot a single meta sample on a figure. This function is NOT used to generate a figure for the paper, but instead
    the function `plot_meta_sample_multitime` is used for that. This function provides more information though.
    For instance, it can draw the graphs, the inferred edges and the z distribution of the context.
    """

    if people_count is None:
        people_count = len(permutation)

    target_future_prediction_mean = None
    if predictions != None:
        target_future_prediction_mean = predictions.stochastic.mean[which_z_sample, :, :, 0, :].detach().numpy()

    # Create the figure and axes
    row_count = max(context_size, target_size)
    fig, ax = plt.subplots(row_count, 3 + 1, figsize=(10, 10))

    # Turn off the axes for the first column
    for i in range(row_count):
        ax[i, 0].axis('off')

    # Draw the ground truth graph in column 0
    underlying_graph = create_graph(people_count, permutation, [], [], True)
    pos = nx.circular_layout(underlying_graph)
    colors = nx.get_edge_attributes(underlying_graph, 'color').values()
    nx.draw(underlying_graph, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[row_count // 2, 0], edge_color=colors)
    ax[row_count // 2, 0].set_title('Ground truth')

    # Draw the context graphs
    for i in range(context_size):
        context_sequence_observed = np.array(meta_sample.context.observed[:, i, 0, :])
        context_sequence_future = np.array(meta_sample.context.future[:, i, 0, :])

        # Decode the context sequence (since it is 1-hot encoded)
        context_sequence_observed = np.argmax(context_sequence_observed, axis=-1)
        context_sequence_future = np.argmax(context_sequence_future, axis=-1)

        # Plot the context_sequence_observed and context_sequence_future as labels
        if plot_order_as_labels:
            ax[i, 1].set_title('Observed: ' + str(context_sequence_observed) + ", Future: " + str(context_sequence_future))

        # Draw the context graph
        G_context = create_graph(people_count, context_sequence_observed, context_sequence_future, [], False)
        pos = nx.circular_layout(G_context)
        colors = nx.get_edge_attributes(G_context, 'color').values()
        nx.draw(G_context, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[i, 1], edge_color=colors)

    # calculate what we can infer from the context alone
    known_edges_context = set()
    for i in range(context_size):
        context_sequence_observed = np.array(meta_sample.context.observed[:, i, 0, :])
        context_sequence_observed = np.argmax(context_sequence_observed, axis=-1)
        for j in range(len(context_sequence_observed) - 1):
            known_edges_context.add((context_sequence_observed[j], context_sequence_observed[j + 1]))

    # calculate what we can infer from the context and the target, for each target point separately
    known_edges_target = [] # an array of sets
    for i in range(target_size):
        known_edges_target.append(set(known_edges_context))
        target_sequence_observed = np.array(meta_sample.target.observed[:, i, 0, :])
        target_sequence_observed = np.argmax(target_sequence_observed, axis=-1)
        for j in range(len(target_sequence_observed) - 1):
            known_edges_target[i].add((target_sequence_observed[j], target_sequence_observed[j + 1]))

    if draw_inferred_edges:
        # Draw the inferred graph for context
        G_inferred_context = nx.DiGraph()
        for i in range(people_count):
            G_inferred_context.add_node(i, label=i)
        for edge in known_edges_context:
            G_inferred_context.add_edge(edge[0], edge[1], color='red')
        pos = nx.circular_layout(G_inferred_context)
        colors = nx.get_edge_attributes(G_inferred_context, 'color').values()
        nx.draw(G_inferred_context, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[-1, 1], edge_color=colors)

        # Draw the inferred graph for the targets
        for i in range(target_size):
            G_inferred_target = nx.DiGraph()
            for j in range(people_count):
                G_inferred_target.add_node(j, label=j)
            for edge in known_edges_target[i]:
                G_inferred_target.add_edge(edge[0], edge[1], color='red')
            pos = nx.circular_layout(G_inferred_target)
            colors = nx.get_edge_attributes(G_inferred_target, 'color').values()
            nx.draw(G_inferred_target, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[i, 3], edge_color=colors)

    if plot_z_distribution:
        # Draw the z distribution
        z_distribution_mean = np.reshape(predictions[0].posteriors.q_context.mean.detach().numpy(), [-1])
        z_distribution_std = np.reshape(predictions[0].posteriors.q_context.stddev.detach().numpy(), [-1])

        z_values = np.linspace(-3, 3, 100)
        z_pdf = np.exp(-0.5 * (z_values - z_distribution_mean) ** 2 / z_distribution_std ** 2) / np.sqrt(2 * np.pi * z_distribution_std ** 2)
        ax[-1, 0].plot(z_values, z_pdf)
        ax[-1, 0].set_title('Z distribution')

        # Add axis ticks
        ax[-1, 0].set_xticks(np.linspace(-3, 3, 7))
        ax[-1, 0].set_xticklabels(np.round(np.linspace(-3, 3, 7), 2))
        ax[-1, 0].axis('on')

    # Draw the target graphs
    for i in range(target_size):
        target_sequence_observed = meta_sample.target.observed[:, i, 0, :]
        target_sequence_future = meta_sample.target.future[:, i, 0, :]
        if target_future_prediction_mean is not None:
            target_sequence_prediction = target_future_prediction_mean[:, i, :]
        else:
            target_sequence_prediction = []

        # Decode the target sequence (since it is 1-hot encoded)
        target_sequence_observed = np.argmax(target_sequence_observed, axis=-1)
        target_sequence_future = np.argmax(target_sequence_future, axis=-1)

        if plot_order_as_labels:
            if target_future_prediction_mean is not None:

                if target_sequence_prediction.shape[0] == 1:
                    target_sequence_prediction = np.squeeze(target_sequence_prediction, axis=0)

                else:
                    target_sequence_prediction = np.argmax(target_sequence_prediction, axis=-1)
                    title += ", Predicted: " + str(np.array(target_sequence_prediction))

            ax[i, 2].set_title(title)

        # Draw the target graph
        G_target = create_graph(people_count, target_sequence_observed, target_sequence_future, target_sequence_prediction, False)
        pos = nx.circular_layout(G_target)
        colors = nx.get_edge_attributes(G_target, 'color').values()
        nx.draw(G_target, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[i, 2], edge_color=colors)

    plt.show()
