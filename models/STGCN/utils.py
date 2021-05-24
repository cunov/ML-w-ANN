import os
import zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from data_prep.prep import load_data, collapse_data


def load_metr_la_data():
    if (not os.path.isfile("models/STGCN/data/adj_mat.npy")
            or not os.path.isfile("models/STGCN/data/node_values.npy")):
        with zipfile.ZipFile("models/STGCN/data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("models/STGCN/data/")

    A = np.load("models/STGCN/data/adj_mat.npy")
    X = np.load("models/STGCN/data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


def load_I_35_data(direction=0, features=[1 ,2], extra_feature='none'):
    X, A, sensor_enums, time_enums, direction_enums = load_data('data_prep/i35_2019')

    X[0], X[1] = collapse_data(X)

    X = X[direction]
    A = A[direction]

    X = np.moveaxis(X, 0, 2)
    X = X[:, features, :]  # Select features

    if extra_feature != 'none':
        # Add periodic feature to help periodic learning (compare with time_2_vec)
        periodic = np.zeros([X.shape[0], 1, X.shape[2]])
        triangle = np.linspace(0, 1, 288)
        triangle = np.tile(triangle, int(X.shape[2] / 288) + 1)

        if extra_feature == 'triangle':
            periodic[:, 0, :] = triangle[:X.shape[2]]
        elif extra_feature == 'sin':
            sin = np.sin(triangle * (2 * np.pi)-np.pi/2) / 2 + 0.5
            periodic[:, 0, :] = sin[:X.shape[2]]
            periodic_tri = periodic.copy()
            periodic_tri[:, 0, :] = triangle[:X.shape[2]]

        X = np.concatenate((X, periodic), axis=1)

    A += np.eye(A.shape[0])  # Add diagonal to allow graph conv

    X[X==-1] = 0 # Remove -1 from data

    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    X = X.astype(np.float32)
    A = A.astype(np.float32)

    # plot
    if False:
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                 '#17becf']
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(20, 5)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 20
        plt.rcParams['savefig.dpi'] = 500

        plt.plot(X[10, 0, 0:288*7], color=color[0])
        plt.text(-300, np.average(X[10, 0, 0:288*7]), 'Speed')

        plt.plot(X[10, 1, 0:288 * 7]+4, color=color[1])
        plt.text(-300, np.average(X[10, 1, 0:288 * 7]+4), 'Volume')

        plt.plot(periodic_tri[10, 0, 0:288 * 7] + 8, color=color[2])
        plt.text(-300, np.average(periodic_tri[10, 0, 0:288 * 7])+8, 'Triangle')

        plt.plot(X[10, 2, 0:288 * 7]+12, color=color[3])
        plt.text(-300, np.average(X[10, 2, 0:288 * 7])+12, 'Sine')

        plt.xlim(-350, 288 * 7)
        plt.yticks([])
        ticks = np.arange(0, 288*7)
        times = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        freq = 288
        plt.xticks(ticks[::freq]+int(288/2), times)
        plt.xticks(rotation=45)
        plt.show()

    # Zoom in
    if False:
        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                 '#17becf']
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(20, 5)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 20
        plt.rcParams['savefig.dpi'] = 500
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

        plt.plot(X[10, 0, 0:int(288/2)], '.', color=color[0])
        #plt.text(-300, np.average(X[10, 0, 0:288/2]), 'Speed')

        plt.plot(X[10, 1, 0:int(288/2)]+4,'.', color=color[1])
        #plt.text(-300, np.average(X[10, 1, 0:288/2]+4), 'Volume')

        plt.plot(periodic_tri[10, 0, 0:int(288/2)] + 8,'.', color=color[2])
        #plt.text(-300, np.average(periodic_tri[10, 0, 0:288/2])+8, 'Triangle')

        plt.plot(X[10, 2, 0:int(288/2)]+12,'.', color=color[3])
        #plt.text(-300, np.average(X[10, 2, 0:288/2])+12, 'Sine')

        plt.xlim(0, int(288/2))
        plt.yticks([])
        ticks = np.arange(0, int(288/2))
        times = ['24:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00']
        freq = 12
        plt.xticks(ticks[::freq], times)
        plt.xticks(rotation=45)
        plt.show()

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))
