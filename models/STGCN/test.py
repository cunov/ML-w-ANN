import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, load_I_35_data

num_timesteps_input = 12
num_timesteps_output = 3

epochs = 1000
batch_size = 50

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
args.enable_cuda = True
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)

    A, X, means, stds = load_I_35_data()

    split_line1 = int(X.shape[2] * 0.9)
    split_line2 = int(X.shape[2] * 0.95)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    net.load_state_dict(torch.load('models/STGCN/state_dict', map_location=args.device))

    test_data = test_input[np.arange(0, 5240, 3), :, :]
    test_target = test_target[np.arange(0, 5240, 3), :]

    #test_data = torch.from_numpy(test_data)
    test_data = test_data.to(device=args.device)

    with torch.no_grad():
        net.eval()
        out = net(A_wave, test_data)

    out = out.detach()
    out = out.cpu()
    out = out.numpy()
    out = np.moveaxis(out, 1, 2)
    out = out.reshape(out.shape[0]*out.shape[1], out.shape[2])

    test_target = test_target.detach()
    test_target = test_target.cpu()
    test_target = test_target.numpy()
    test_target = np.moveaxis(test_target, 1, 2)
    test_target = test_target.reshape(test_target.shape[0] * test_target.shape[1], test_target.shape[2])


    plt.plot(np.average(test_target,axis=1))
    plt.plot(np.average(out,axis=1))
    plt.show()

    x = 1
