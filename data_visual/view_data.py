from data_prep import prep
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats as stats
from tools.tools import key_by_val, invert_numeric_dict_to_list, collapse_data


def plot_grid(distance_matrix, sensor_enums):

    labels_north = invert_numeric_dict_to_list(sensor_enums[0])
    labels_south = invert_numeric_dict_to_list(sensor_enums[1])
    labels_north_dict = {}
    labels_south_dict = {}
    for i in range(len(labels_north)):
        labels_north_dict[i] = labels_north[i]
    for i in range(len(labels_south)):
        labels_south_dict[i] = labels_south[i]

    G1 = nx.from_numpy_matrix(distance_matrix[0][:, :])
    G2 = nx.from_numpy_matrix(distance_matrix[1][:, :])

    elarge1 = [(u, v) for (u, v, d) in G1.edges(data=True)]
    elarge2 = [(u, v) for (u, v, d) in G2.edges(data=True)]

    pos1 = nx.kamada_kawai_layout(G1)
    pos2 = nx.kamada_kawai_layout(G2)

    for k, v in pos2.items():
        # Shift the x values of every node by 10 to the right
        v[0] = v[0] + 0.5

    nx.draw_networkx_nodes(G1, pos=pos1, node_size=30, node_color='b')
    nx.draw_networkx_edges(G1, pos1, edgelist=elarge1, width=1, style='solid')
    nx.draw_networkx_labels(G1, pos1, labels_north_dict, font_size=5)

    nx.draw_networkx_nodes(G2, pos=pos2, node_size=50)
    nx.draw_networkx_edges(G2, pos2, edgelist=elarge2, width=1)
    nx.draw_networkx_labels(G2, pos2, labels_south_dict, font_size=5)

    plt.show()  # display
    plt.draw()


def check_dist_mat(distance_matrix, sensor_enums):

    connected_nodes = 0
    for dir in range(2):
        for i in range(len(distance_matrix[dir])):
            row = distance_matrix[dir][i, :]
            num_connections = len(row[row!=0])
            if num_connections == 0:
                print('Unconnected node: {}'.format(key_by_val(sensor_enums[dir], i)))
            else:
                connected_nodes +=1


    fig, ax = plt.subplots(1, 2)

    cmap = plt.cm.get_cmap('hot')
    #cmap = cmap.reversed()

    ax[0].matshow(distance_matrix[0], cmap=cmap)
    ax[1].matshow(distance_matrix[1], cmap=cmap)
    #ax[2].matshow(np.abs(distance_matrix[0]-distance_matrix[1]), cmap=cmap)

    plt.show()


def plot_single_sensor_evolution(data, sensor_enums, time_enums, direction_enums, sensor):

    time_stamps = max(time_enums.values()) + 1
    days = 365

    collapsed_data_north, collapsed_data_south = collapse_data(data)
    collapsed_data_north = collapsed_data_north[:, sensor, :]
    collapsed_data_south = collapsed_data_south[:, sensor, :]

    fig, ax = plt.subplots()

    ax.plot(collapsed_data_north[:, 2], label=key_by_val(direction_enums, 0) + ' Avg: {:.2f}'.format(np.average(collapsed_data_north[:, 1])))
    #if np.sum(collapsed_data_north[:, 3]) > 0:
    #    ax.fill_between(np.arange(int(days*time_stamps)), collapsed_data_north[:, 1] - collapsed_data_north[:, 3],
    #                    collapsed_data_north[:, 1] + collapsed_data_north[:, 3], alpha=0.2)

    #ax.plot(collapsed_data_south[:, 1], label=key_by_val(direction_enums, 1) + ' Avg: {:.2f}'.format(np.average(collapsed_data_south[:, 1])))
    #if np.sum(collapsed_data_south[:, 3]) > 0:
    #    ax.fill_between(np.arange(int(days * time_stamps)), collapsed_data_south[:, 1] - collapsed_data_south[:, 3],
    #                collapsed_data_south[:, 1] + collapsed_data_south[:, 3], alpha=0.2)

    ax.set_title(key_by_val(sensor_enums[0], sensor))
    ax.set_xticks(np.arange(0, len(collapsed_data_south[:, 1]), 288))

    plt.legend()
    plt.show()


def plot_sensor_distributions(data, sensor_enums, time_enums, direction_enums):

    collapsed_data_north, collapsed_data_south = collapse_data(data)

    avg = [None] * 2
    avg[0] = np.average(collapsed_data_north, axis=0)
    avg[1] = np.average(collapsed_data_south, axis=0)

    fig, ax = plt.subplots(1, 2)
    for dir in range(2):
        sensors = len(sensor_enums[dir])
        for i in range(sensors):
            mean = avg[dir][i, 1]
            std = avg[dir][i, 3]

            if std == 0 or std == -1:
                print('The sensor {}, has constant values in the {} direction'.format(key_by_val(sensor_enums[dir], i), key_by_val(direction_enums, dir)))
                continue

            x = np.linspace(mean-4*std, mean+4*std, 100)

            ax[dir].plot(x, stats.norm.pdf(x, mean, std))

            ax[dir].set_ylim([0, 1])

    plt.show()


if __name__ == '__main__':
    data, distance_matrix, sensor_enums, time_enums, direction_enums = prep.load_data('data_prep/i35_2019')
    check_dist_mat(distance_matrix, sensor_enums)
    plot_grid(distance_matrix, sensor_enums)
    plot_single_sensor_evolution(data, sensor_enums, time_enums, direction_enums, 0)
    plot_sensor_distributions(data, sensor_enums, time_enums, direction_enums)
