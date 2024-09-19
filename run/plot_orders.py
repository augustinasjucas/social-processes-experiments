import numpy as np
import matplotlib.pyplot as plt
from data.types import DataSplit
import networkx as nx

def create_graph(n,
                 observed_permutation,
                 future_permutation,
                 predicted_permutation,
                 use_loop=False):
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
    Given a meta sample, creates a set of edges that can be deduced with full certainty.
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
    Given a meta sample, creates a set of edges that can be deduced with full certainty.
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
    Draws a heatmap of the observed, future and predicted sequences.
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

    # Create observed matrix
    for i in range(observed_length):
        person = np.argmax(observed[:, i])
        observed_matrix[person, i] = 1
    ax.matshow(observed_matrix, cmap='Blues', aspect='1')

    # Go over each future timestep and draw a rectangle around the cell
    for i in range(future_length):
        # ax.matshow(np.zeros(future.shape))
        future_person = np.argmax(future[:, i])
        ax.add_patch(plt.Rectangle((observed_length + i - 0.5, future_person - 0.5), 1, 1, fill=None, edgecolor='red'))

    # If we have predicted values, matshow them as well
    if predicted is not None:
        # Create predicted matrix
        if predicted is not None:
            for i in range(future_length):
                predicted_matrix[:, observed_length + i] = predicted[:, i]
        ax.matshow(predicted_matrix, cmap='Greens', aspect='1', )

        # Show the predicted values as text
        for (i, j), z in np.ndenumerate(predicted_matrix):
            if j >= observed_length:
                ax.text(j, i, '{:.0f}'.format(z*100) + "%", ha='center', va='center', fontsize=4, color='black')

    # Draw grid
    ax.set_xticks([x - 0.5 for x in range(1, width)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, height)], minor=True)
    ax.grid(c='indigo', ls=':', lw='0.4', which='minor')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='y', labelsize=6)

def draw_inferred_graph(
        ax,
        known_edges,
        meta_sample
):
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
    # Draw the q(z | C)
    z_distribution_mean = np.reshape(predicted.posteriors.q_context.mean.detach().numpy(), [-1])
    z_distribution_std = np.reshape(predicted.posteriors.q_context.stddev.detach().numpy(), [-1])

    # On ax_text, write the mean and std (rounded to 4 decimals)
    ax_text.axis('off')
    print(np.round(z_distribution_mean, 4))
    ax_text.text(0.5, 0.5, "Mean: " + str(np.round(z_distribution_mean, 4))
                 # + "\n" + "Std: " + str(np.round(z_distribution_std, 4))
                 ,
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
    target_size_capped = min(target_size, 5)
    context_size_capped = min(context_size, 5)

    target_indices_to_display = list(range(target_size))
    if target_size_capped != target_size:
        target_indices_to_display = list(set(range(0, context_size_capped)).union(set(range(target_size - target_size_capped, target_size))))
        target_size_capped = len(target_indices_to_display)


    # to_del = []
    # for i in target_indices_to_display:
    #     if i >= target_size-context_size:
    #         # Remove the element from target indices
    #         to_del.append(i)
    # for i in to_del:
    #     target_indices_to_display.remove(i)
    #
    # print("target indices to display:", target_indices_to_display,  target_size-context_size)

    column_count = 4
    if predicted is not None:
        column_count += len(predicted) - 1

    # Create an ax with 2 columns and max(context_size + 2, target_size) rows
    fig, ax = plt.subplots(max(target_size_capped, context_size_capped, 4), column_count, figsize=(10, 10))

    # Draw the context graphs on 1st column
    for i in range(context_size_capped):
        draw_heatmap(ax[i, 0], meta_sample.context.observed[:, i, 0, :], meta_sample.context.future[:, i, 0, :])

    # Draw the inferred known ordering graph for the context
    inferred_edges = infer_edges(context_size, meta_sample)
    draw_inferred_graph(ax[2, 1], inferred_edges, meta_sample)
    ax[2, 1].set_title("What can be inferred from the context")

    # Draw the inferred know orderings for each target sample
    for i, target_index in enumerate(target_indices_to_display):
        inferred_edges = infer_edges_for_target(context_size, meta_sample, meta_sample.target.observed[:, target_index, 0, :])
        draw_inferred_graph(ax[i, -1], inferred_edges, meta_sample)
    ax[0, -1].set_title("What can be inferred for targets", fontsize=8)

    # Draw q of z given z
    if predicted is not None:
        draw_q_of_z(ax[0, 1], ax[1, 1], predicted[0])


    # Draw the target graphs on 2nd column
    if predicted is None:
        predicted_for_this_person = None
        for i, target_index in enumerate(target_indices_to_display):
            draw_heatmap(ax[i, 2], meta_sample.target.observed[:, target_index, 0, :], meta_sample.target.future[:, target_index, 0, :], predicted_for_this_person)
    else:
        for j in range(len(predicted)):
            z_sample_string = "z sample " + str(j)
            if final_result_is_random and j == len(predicted) - 1:
                z_sample_string = "Random z sample"

            ax[0, j + 2].set_title("Target (" + str(target_size_capped) + "/" + str(target_size) + "), " + z_sample_string, fontsize=8)
            for i, target_index in enumerate(target_indices_to_display):
                if predictions_are_normal:
                    predicted_for_this_person = (predicted[j].stochastic.mean[0, :, :, 0, :])[:, target_index, :]
                else:
                    print("PREDICTIONS SHAPE IS", predicted[j].stochastic.probs.shape)
                    predicted_for_this_person = predicted[j].stochastic.probs[0, :, :, 0, :][:, target_index, :]
                draw_heatmap(ax[i, 2 + j], meta_sample.target.observed[:, target_index, 0, :], meta_sample.target.future[:, target_index, 0, :], predicted_for_this_person)

    # Set names for context and target columns
    ax[0, 0].set_title("Context" + " (" + str(context_size_capped) + "/" + str(context_size) + ")")

    # Add a line between context and target
    # line = plt.Line2D((.3, .3), (0, 1), color="k", linewidth=3)
    # fig.add_artist(line)

    # Delete unused axes
    axes_to_rem = []
    axes_to_rem.append(ax[context_size_capped:, 0])
    axes_to_rem.append(ax[target_size_capped:, 2])
    axes_to_rem.append(ax[3:, 1])

    if predicted is None:
        axes_to_rem.append([ax[0, 1], ax[1, 1]])

    for lst in axes_to_rem:
        for x in lst:
            fig.delaxes(x)

    # Set top and bottom properties
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0)
    plt.show()


def plot_losses(
        losses,
        posteriors,
        trained_models,
        test_datasets,
        args
):
    some_colors = ['brown', 'orange', 'green', 'red', 'pink']
    model_cnt = len(losses)
    dataset_cnt = len(losses[0])
    row_count = (model_cnt + 1) if args.my_plot_posteriors else 1

    # Create a 2d grid of plots dataset_cnt x 1
    fig, ax = plt.subplots(row_count, max(dataset_cnt, 2), figsize=(10, 10))

    for i in range(dataset_cnt):
        index = i
        if args.my_plot_posteriors:
            index = (0, i)

        max_loss = 0
        # Set title for this plot to be the name of the dataset
        ax[index].set_title("Tested on the " + test_datasets[i].__name__)
        for j in range(model_cnt):

            # Plot a bar of height losses[i][j]. Set title for this bar to be the name of the model
            ax[index].bar(j, losses[j][i], label="Trained on " + trained_models[j][0] + " dataset", color=some_colors[j])
            max_loss = max(max_loss, losses[j][i])

            # show legend
            ax[index].legend()

            # set axes
            ax[index].set_ylim(bottom=0)
            ax[index].set_xlim(-1.5, 0.5 + model_cnt)
            ax[index].set_xticks([])
            ax[index].set_ylabel("Loss")
            ax[index].set_xlabel("Model")


            # Each posteriors[j][i] is a list of (mean, variance) of Normal Distributions. Plot first 10 of those
            # in the axis (j+1, i), with the title "ten distributions q(z|C) with model trained on <model name>"
            if args.my_plot_posteriors:
                mn = np.inf
                mx = -np.inf

                slices = ""

                for k, (mean, variance) in enumerate(posteriors[j][i][:10]):
                    std = np.sqrt(variance)

                    if len(mean.shape) == 2 and mean.shape[1] != 1:
                        std = std[:, 0]
                        mean = mean[:, 0]
                        variance = variance[:, 0]
                        slices = "slices of "

                    z_values = np.linspace(mean-4*std, mean+4*std, 100)
                    mn = min(mn, mean-4.1*std)
                    mx = max(mx, mean+4.1*std)
                    z_pdf = np.exp(-0.5 * (z_values - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)
                    z_pdf = np.reshape(z_pdf, (100,))
                    z_values = np.reshape(z_values, (100,))

                    ax[j + 1, i].plot(z_values, z_pdf, color=some_colors[j])

                ax[j + 1, i].set_title("Ten " + slices + "q(z|C) distributions of meta-samples with model trained on " + trained_models[j][0] + " dataset", fontsize=10)

                ax[j + 1, i].set_xlim(mn, mx)
                ax[j+1, i].set_ylim(bottom=0)
        ax[index].set_ylim(top=max_loss*1.1)
    # Set wspace, etc
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.036, wspace=0.2, hspace=0.368)

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

    Args:
        meta_sample:
        permutation: np array of shape (n)

    Returns:

    """
    if people_count is None:
        people_count = len(permutation)

    # (nz_samples, target_len, batch_size, npeople, data_dim)
    # print("prediction has shape", predictions.stochastic.mean.shape)
    target_future_prediction_mean = None
    if predictions != None:
        target_future_prediction_mean = predictions.stochastic.mean[which_z_sample, :, :, 0, :].detach().numpy()
    #     target_sequence_prediction = target_future_prediction_mean[:, i, :]

    # Leave last row for inferred context edges and last column for inferred target edges
    row_count = max(context_size, target_size)
    # Create a plot with 4 columns and max(context_size, target_size) rows
    fig, ax = plt.subplots(row_count, 3 + 1, figsize=(10, 10))

    # Merge column 0
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
        # Check if the known_edges_target[i] has size equal to n-1, and then we can infer the last edge
        # if len(known_edges_target[i]) == len(permutation) - 1:

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
        print("predictions = " , predictions)
        # Plot the distribution as a linspace
        # print("z_distrib mean shape = " + z_distribution_mean.shape)

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

            print("target sequence observed: " + str(np.array(target_sequence_observed)) + ", shape is: " + str(np.array(meta_sample.target.observed.shape)))
            print("target sequqnce future: " + str(np.array(target_sequence_prediction)))

        # Decode the target sequence (since it is 1-hot encoded)
        target_sequence_observed = np.argmax(target_sequence_observed, axis=-1)
        target_sequence_future = np.argmax(target_sequence_future, axis=-1)

        if plot_order_as_labels:
            title = 'Observed: ' + str(np.array(target_sequence_observed)) + ", Future: " + str(np.array(target_sequence_future))

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

"""
    1. If I do clockwise-anticlockwise, and I put more than 1 timestep as "observed", the model can infer the direction without the context information.
        - Is that a problem? I think yes
        - Solution: give as context 'some number' of timesteps, but for target prediction give only 1 (this is what I did). 
        - How much do you dislike this solution? It is not ideal, but probably kinda fine.
z        a) well maybe, we could say that in the "dominant" experiment, when we give partial context (of not all people but only some of them), that
        the z sample is a sample of what happens with the "missing" people. But that is only for test time. In train time, q(z | C) is useless and probably it is 
        even implicitly pushed to collapse by the way it is trained. 
"""

"""r

"""