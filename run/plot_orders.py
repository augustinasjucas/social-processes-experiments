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
        G.add_node(i, label=i)

    # Convert to numpy arrays
    observed_permutation = np.array(observed_permutation)
    future_permutation = np.array(future_permutation)

    observed_color = 'blue'
    future_color = 'lightblue'

    for i in range(len(observed_permutation) - 1):
        G.add_edge(observed_permutation[i], observed_permutation[i + 1], color=observed_color)

    if use_loop:
        G.add_edge(observed_permutation[-1], observed_permutation[0], color=observed_color)

    # Add the last element of the observed permutation to the start of future permutation
    future_permutation = np.insert(future_permutation, 0, observed_permutation[-1])
    for i in range(len(future_permutation) - 1):
        G.add_edge(future_permutation[i], future_permutation[i + 1], color=future_color)

    # Add the last element of the observed_permutation permutation to the start of predicted permutation
    predicted_permutation = np.insert(predicted_permutation, 0, observed_permutation[-1])
    for i in range(len(predicted_permutation) - 1):
        G.add_edge(predicted_permutation[i], predicted_permutation[i + 1], color='green')

    return G

def plot_meta_sample(
        meta_sample: DataSplit,
        permutation: np.array,
        context_size,
        target_size,
        predictions=None,
        which_z_sample=0
):
    """

    Args:
        meta_sample:
        permutation: np array of shape (n)

    Returns:

    """
    # (nz_samples, target_len, batch_size, npeople, data_dim)
    # print("prediction has shape", predictions.stochastic.mean.shape)
    target_future_prediction_mean = None
    if predictions != None:
        print("=== predictions stochstic mean shape: ", predictions.stochastic.mean.shape)
        target_future_prediction_mean = predictions.stochastic.mean[which_z_sample, :, :, 0, :].detach().numpy()

    # Leave last row for inferred context edges and last column for inferred target edges
    row_count = max(context_size, target_size)
    # Create a plot with 4 columns and max(context_size, target_size) rows
    fig, ax = plt.subplots(row_count, 3 + 1, figsize=(10, 10))

    # Merge column 0
    for i in range(row_count):
        ax[i, 0].axis('off')

    # Draw the ground truth graph in column 0
    underlying_graph = create_graph(len(permutation), permutation, [], [], True)
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

        # Draw the context graph
        G_context = create_graph(len(permutation), context_sequence_observed, context_sequence_future, [], False)
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

        # # Check if the known_edges_context has size equal to n-1, and then we can infer the last edge
        # if len(known_edges_context) == len(permutation) - 1:
        #     for j in range(len(permutation)):
        #         if (j, j + 1) not in known_edges_context:
        #             known_edges_context.add((j, j + 1))
        #             break

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


    # Draw the inferred graph for context
    G_inferred_context = nx.DiGraph()
    for i in range(len(permutation)):
        G_inferred_context.add_node(i, label=i)
    for edge in known_edges_context:
        G_inferred_context.add_edge(edge[0], edge[1], color='red')
    pos = nx.circular_layout(G_inferred_context)
    colors = nx.get_edge_attributes(G_inferred_context, 'color').values()
    nx.draw(G_inferred_context, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[-1, 1], edge_color=colors)

    # Draw the inferred graph for the targets
    for i in range(target_size):
        G_inferred_target = nx.DiGraph()
        for j in range(len(permutation)):
            G_inferred_target.add_node(j, label=j)
        for edge in known_edges_target[i]:
            G_inferred_target.add_edge(edge[0], edge[1], color='red')
        pos = nx.circular_layout(G_inferred_target)
        colors = nx.get_edge_attributes(G_inferred_target, 'color').values()
        nx.draw(G_inferred_target, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[i, 3], edge_color=colors)


    # Draw the target graphs
    for i in range(target_size):
        target_sequence_observed = meta_sample.target.observed[:, i, 0, :]
        target_sequence_future = meta_sample.target.future[:, i, 0, :]
        if target_future_prediction_mean is not None:
            target_sequence_prediction = target_future_prediction_mean[:, i, :]
        else:
            target_sequence_prediction = []

            print("target sequence observed: " + str(target_sequence_observed) + ", shape is: " + str(meta_sample.target.observed.shape))
            print("target sequqnce future: " + str(target_sequence_prediction))

        # Decode the target sequence (since it is 1-hot encoded)
        target_sequence_observed = np.argmax(target_sequence_observed, axis=-1)
        target_sequence_future = np.argmax(target_sequence_future, axis=-1)
        if target_future_prediction_mean is not None:
            target_sequence_prediction = np.argmax(target_sequence_prediction, axis=-1)

        # Draw the target graph
        G_target = create_graph(len(permutation), target_sequence_observed, target_sequence_future, target_sequence_prediction, False)
        pos = nx.circular_layout(G_target)
        colors = nx.get_edge_attributes(G_target, 'color').values()
        nx.draw(G_target, pos, with_labels=True, node_size=300, node_color='skyblue', ax=ax[i, 2], edge_color=colors)

    plt.show()

