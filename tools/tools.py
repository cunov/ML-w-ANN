import numpy as np


def invert_numeric_dict_to_list(d):
    l = [None] * len(d)
    for key in d:
        l[d[key]] = key

    return l


def key_by_val(dict, val):
    return list(dict.keys())[list(dict.values()).index(val)]


def remove_items_from_numeric_dict(dict, items):
    remove_keys = [key_by_val(dict, x) for x in items]
    for r_key in remove_keys:
        for key in dict:
            if dict[key] > dict[r_key]:
                dict[key] -= 1

    for r_key in remove_keys:
        dict.pop(r_key, None)

    return dict


def num_lines(txt_file):
    file = open(txt_file, "r")
    nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
    line_count = len(nonempty_lines)
    file.close()
    return line_count


def collapse_data(data):
    num_sensors_north = data[0].shape[0]
    num_sensors_south = data[1].shape[0]
    num_days = data[0].shape[1]
    num_time_stamps = data[0].shape[2]

    collapsed_data_north = np.zeros([int(num_days * num_time_stamps), num_sensors_north, 4])
    collapsed_data_south = np.zeros([int(num_days * num_time_stamps), num_sensors_south, 4])
    iter = 0
    for d in range(num_days):
        for t in range(num_time_stamps):
            collapsed_data_north[iter, :, :] = data[0][:, d, t, :]
            collapsed_data_south[iter, :, :] = data[1][:, d, t, :]
            iter += 1

    return collapsed_data_north, collapsed_data_south


def divide_data(data, frac_train, frac_val, frac_test):
    num_samples = data.shape[0]

    tot_frac = frac_train + frac_val + frac_test
    splt_point_1 = int(frac_train / tot_frac * num_samples)
    splt_point_2 = int((frac_train + frac_val) / tot_frac * num_samples)

    train_data = data[:splt_point_1]
    val_data = data[splt_point_1:splt_point_2]
    test_data = data[splt_point_2:]

    return train_data, val_data, test_data


def split_data(data, input_sequence_length, output_sequence_length, num_samples_to_be_drawn, features=[0, 1, 2, 3]):
    num_samples = data.shape[0]
    num_sensors = data.shape[1]
    num_features = len(features)

    num_possible_splits = num_samples - (input_sequence_length + output_sequence_length - 1)

    if num_possible_splits < num_samples_to_be_drawn:
        print(
            'To many requested samples: {}, must be less than total possible number: {}'.format(num_samples_to_be_drawn,
                                                                                                num_possible_splits))
        return

    split_data_input = np.zeros([num_samples_to_be_drawn, input_sequence_length, num_sensors, num_features], dtype='e')
    split_data_output = np.zeros([num_samples_to_be_drawn, output_sequence_length, num_sensors, num_features],
                                 dtype='e')

    chosen = np.random.choice(np.arange(num_possible_splits), size=num_samples_to_be_drawn, replace=False)

    for indx, i in enumerate(chosen):
        split_data_input[indx] = data[i:(i + input_sequence_length), :, features]
        split_data_output[indx] = data[(i + input_sequence_length):(i + input_sequence_length + output_sequence_length),
                                  :, features]

    split_data_input = np.squeeze(split_data_input)
    split_data_output = np.squeeze(split_data_output)

    return split_data_input, split_data_output


def non_overlapping_moving_window_average(array, window_size):
    """
    Performs a non overlapping moving window average
    :param array: 1D data vector
    :param window_size: the number of points to be averaged for each new data point
    :return: 1D array of size largest_divisor
    """
    largest_divisor = int(len(array) / window_size) * window_size
    return array[:largest_divisor].reshape(-1, window_size).mean(1)


def check_for_no_connections(dist_mat):
    for d in range(2):
        num_sensors = dist_mat[d].shape[0]
        for sensor in range(num_sensors):
            if len(np.where(dist_mat[d][sensor, :] != 0)[0]) == 0:
                print(sensor)
