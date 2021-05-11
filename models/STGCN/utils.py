import os
import zipfile
import numpy as np
import torch
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


def load_I_35_data(direction=0, features=[1 ,2]):
    X, A, sensor_enums, time_enums, direction_enums = load_data('data_prep/i35_2019')

    X[0], X[1] = collapse_data(X)

    X = X[direction]
    A = A[direction]

    X = np.moveaxis(X, 0, 2)
    X = X[:, features, :]  # Select features

    # Add periodic feature to help periodic learning (compare with time_2_vec)
    periodic = np.zeros([X.shape[0], 1, X.shape[2]])
    triangle = np.linspace(0, 1, 288)
    triangle = np.tile(triangle, int(X.shape[2]/288)+1)
    periodic[:, 0, :] = triangle[:X.shape[2]]

    X = np.concatenate((X, periodic), axis=1)

    A += np.eye(A.shape[0])  # Add diagonal to allow graph conv

    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    X = X.astype(np.float32)
    A = A.astype(np.float32)

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
