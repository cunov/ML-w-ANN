import gzip
import os
import argparse
import pickle as pk

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime

from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data, get_normalized_adj, load_I_35_data
from data_prep.prep import load_data
from tools.tools import invert_numeric_dict_to_list

num_timesteps_input = 12
num_timesteps_output = 6

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

def test(file_name):
    torch.manual_seed(7)

    _, _, sensor_enums, time_enums, direction_enums = load_data('data_prep/i35_2019')
    times = invert_numeric_dict_to_list(time_enums)
    times = np.array(times)
    for i in range(len(times)):
        times[i] = datetime.datetime.strptime(times[i], '%I:%M:%S %p').strftime("%H:%M")

    if file_name == 'sin_30' or file_name == 'sin_30_long':
        A, X, means, stds = load_I_35_data(extra_feature='sin')
    elif file_name == 'triangle_30' or file_name == 'triangle_30_long':
        A, X, means, stds = load_I_35_data(extra_feature='triangle')
    elif file_name == 'no_extra_30' or file_name == 'no_extra_30_long':
        A, X, means, stds = load_I_35_data(extra_feature='none')

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

    net.load_state_dict(torch.load('models/STGCN/state_dict_{}'.format(file_name), map_location=args.device))

    test_data = test_input[np.arange(0, test_input.shape[0] - 1, num_timesteps_output), :, :]
    test_target = test_target[np.arange(0, test_input.shape[0] - 1, num_timesteps_output), :]

    test_data = test_data.to(device=args.device)

    with torch.no_grad():
        net.eval()
        out = net(A_wave, test_data)

    out = out.detach()
    out = out.cpu()
    out = out.numpy()
    out = np.moveaxis(out, 1, 2)
    out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])

    test_target = test_target.detach()
    test_target = test_target.cpu()
    test_target = test_target.numpy()
    test_target = np.moveaxis(test_target, 1, 2)
    test_target = test_target.reshape(test_target.shape[0] * test_target.shape[1], test_target.shape[2])

    # Denormalize
    test_target = (test_target * stds[0]) + means[0]
    out = (out * stds[0]) + means[0]

    # Calc test score
    out_avg = test_input.detach().cpu().numpy()[:, :, :, 0]*stds[0]+means[0]
    out_avg = np.average(out_avg, axis=2)[1:]

    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    diff = reject_outliers(np.abs(out-test_target))
    diff_avg = reject_outliers(np.abs(out_avg - test_target))

    print('{} test mae: {}, average mae {}'.format(file_name, np.mean(diff), np.mean(diff_avg)))


    return out, test_target, times


def view_averaged_speed():
    none, _, _ = test('no_extra_30_long')
    triangle, _, _ = test('triangle_30_long')
    sin, target, times = test('sin_30_long')

    none = np.average(none, axis=1)
    triangle = np.average(triangle, axis=1)
    sin = np.average(sin, axis=1)
    target = np.average(target, axis=1)

    fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(10, 7)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams['savefig.dpi'] = 500

    plt.plot(target, label='Target')
    plt.plot(none, label='Normal')
    plt.plot(triangle, label='Triangle')
    plt.plot(sin, label='Sine')

    ticks = np.arange(0, none.shape[0])
    times = np.tile(times, int(ticks.shape[0] / 288) + 1)
    times = times[:ticks.shape[0]]
    freq = 24
    plt.xticks(ticks[::freq], times[::freq])
    plt.xticks(rotation=45)
    plt.ylabel('Speed (mph)')
    plt.legend()
    plt.show()


def view_training():

    def read_loss(file):
        checkpoint_path = "models/STGCN/checkpoints/losses_{}.npy".format(file)
        ret = np.load(checkpoint_path)
        return ret

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(20, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['savefig.dpi'] = 500

    files = ['no_extra_30_long', 'triangle_30_long', 'sin_30_long']
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    label=['Normal', 'Triangle', 'Sine']
    for i, file in enumerate(files):
        out = read_loss(file)
        training_losses, validation_losses, validation_maes = out[0], out[1], out[2]

        if i == 2:
            validation_maes[7] = (validation_maes[6]+validation_maes[8])/2

        print('min mae: {}, {}'.format(file, np.min(validation_maes)))

        plt.plot(validation_maes, color=color[i+1], label=label[i])

    plt.legend()
    plt.ylabel('MAE')
    plt.xlabel('Epochs')
    plt.show()

if __name__ == '__main__':

    view_averaged_speed()
