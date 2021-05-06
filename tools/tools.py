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


def check_for_no_connections(dist_mat):

    for d in range(2):
        num_sensors = dist_mat[d].shape[0]
        for sensor in range(num_sensors):
            if len(np.where(dist_mat[d][sensor, :]!=0)[0]) == 0:
                print(sensor)
