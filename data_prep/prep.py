import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np
from tools.tools import divide_data, split_data, num_lines, collapse_data, remove_items_from_numeric_dict


def read_log_dir(log_dir, save_file):
    num_days = len(os.listdir(log_dir)) - 1
    data = [None] * num_days

    for file_indx, filename in enumerate(os.listdir(log_dir)):
        filename = os.path.join(log_dir, filename)

        if filename.endswith('.txt'):
            data[file_indx] = [None] * num_lines(filename)
            line_iter = 0
            with open(filename) as f:
                for line in f:
                    data[file_indx][line_iter] = [x.strip() for x in line.split(',')]
                    line_iter += 1

    with open(save_file + '_raw_data' '.pkl', 'wb') as filehandle:
        pickle.dump(data, filehandle)


def load_raw_data(save_file):
    with open(save_file + '_raw_data' '.pkl', 'rb') as filehandle:
        raw_data = pickle.load(filehandle)

    return raw_data


def parse_data(save_file):
    raw_data = load_raw_data(save_file)

    # ignore fork near Dallas
    ignore = ['IH-35W_Alvarado', 'IH-35W_Grandview', 'IH-35E_Forreston', 'IH-35W_FM-66', 'IH-35E_Milford',
              'IH-35W_FM-2959', 'IH-35_Hillsboro', 'IH-35_FM-3267', 'IH-35E_EastWye']

    num_sensors = 63 - len(ignore)
    num_days = len(raw_data)
    num_measurements = len(raw_data[0])
    time_segments = 288 # assuming 5 minute time window

    data = np.zeros([2, num_sensors, num_days, time_segments, 4])
    distance_matrix = np.zeros(([2, num_sensors, num_sensors]))

    sensor_enums = {}

    time_enums = {}
    for i in range(time_segments):
        date = raw_data[0][i+1][9].split(' ')
        time = date[1] + ' ' + date[2]
        time_enums[time] = i

    direction_enums = {'Northbound': 0, 'Southbound': 1}

    for day in range(num_days):
        for i in range(1, num_measurements):
            date = raw_data[day][i][9].split(' ')
            time_str = date[1] + ' ' + date[2]

            time_stamp = time_enums[time_str]
            direction = direction_enums[raw_data[day][i][7]]

            try:
                sensor = sensor_enums[raw_data[day][i][0]]
            except KeyError:

                if raw_data[day][i][0] not in ignore:

                    print('Adding sensor {} to sensor enums'.format(raw_data[day][i][0]))

                    if not bool(sensor_enums):
                        sensor_enums[raw_data[day][i][0]] = 0
                    else:
                        sensor_enums[raw_data[day][i][0]] = max(sensor_enums.values()) + 1
                    sensor = sensor_enums[raw_data[day][i][0]]

                else:
                    continue

            data[direction, sensor, day, time_stamp, 0] = raw_data[day][i][10]
            data[direction, sensor, day, time_stamp, 1] = raw_data[day][i][11]
            data[direction, sensor, day, time_stamp, 2] = raw_data[day][i][13]
            data[direction, sensor, day, time_stamp, 3] = raw_data[day][i][14]

            try:
                sensor_out = sensor_enums[raw_data[day][i][1]]
            except KeyError:

                if raw_data[day][i][1] not in ignore:

                    print('Adding sensor {} to sensor enums'.format(raw_data[day][i][1]))

                    if not bool(sensor_enums):
                        sensor_enums[raw_data[day][i][1]] = 0
                    else:
                        sensor_enums[raw_data[day][i][1]] = max(sensor_enums.values()) + 1
                    sensor_out = sensor_enums[raw_data[day][i][1]]

                else:
                    continue

            distance_matrix[direction, sensor, sensor_out] = raw_data[day][i][8]


    # Ensure the distance matrix is symmetric
    distance_matrix[0, :, :] = np.maximum(distance_matrix[0, :, :], distance_matrix[0, :, :].transpose())
    distance_matrix[1, :, :] = np.maximum(distance_matrix[1, :, :], distance_matrix[1, :, :].transpose())

    np.save(save_file + '_parsed_data_north.npy', data[0])
    np.save(save_file + '_parsed_data_south.npy', data[1])

    np.save(save_file + '_distance_matrix_north.npy', distance_matrix[0])
    np.save(save_file + '_distance_matrix_south.npy', distance_matrix[1])

    f = open(save_file + "_sensor_enums.pkl", "wb")
    pickle.dump(sensor_enums, f)
    f.close()

    f = open(save_file + "_time_enums.pkl", "wb")
    pickle.dump(time_enums, f)
    f.close()

    f = open(save_file + "_direction_enums.pkl", "wb")
    pickle.dump(direction_enums, f)
    f.close()


def load_data(save_file):
    with open(save_file + '_parsed_data_north.npy', 'rb') as f:
        data_north = np.load(f)
    with open(save_file + '_parsed_data_south.npy', 'rb') as f:
        data_south = np.load(f)
    with open(save_file + '_distance_matrix_north.npy', 'rb') as f:
        distance_matrix_north = np.load(f)
    with open(save_file + '_distance_matrix_south.npy', 'rb') as f:
        distance_matrix_south = np.load(f)

    distance_matrix = [distance_matrix_north, distance_matrix_south]
    data = [data_north, data_south]

    f = open(save_file+"_sensor_enums.pkl", "rb")
    sensor_enums = pickle.load(f)
    f.close()

    if len(sensor_enums) != 2:
        temp = [sensor_enums, sensor_enums]
        sensor_enums = temp

    f = open(save_file + "_time_enums.pkl", "rb")
    time_enums = pickle.load(f)
    f.close()

    f = open(save_file+"_direction_enums.pkl", "rb")
    direction_enums = pickle.load(f)
    f.close()

    return data, distance_matrix, sensor_enums, time_enums, direction_enums


def remove_constant_sensors(save_file):
    data, distance_matrix, sensor_enums, time_enums, direction_enums = load_data('data_prep/i35_2019')
    collapsed_data_north, collapsed_data_south = collapse_data(data)
    sensor_enums_new = sensor_enums.copy()

    avg = [None] * 2
    avg[0] = np.average(collapsed_data_north, axis=0)
    avg[1] = np.average(collapsed_data_south, axis=0)

    removed_indices = [[], []]
    for d in range(2):
        sensors = distance_matrix[d].shape[0]
        removed_items = 0
        i = 0
        while i < sensors:
            std = avg[d][i, 3]
            # Remove this sensor in this direction as it is constant
            if std == -1 or std == 0:
                removed_indices[d].append(i+removed_items)
                removed_items += 1

                # Connect two neighboring nodes, assuming only two connections
                summed_dist = np.sum(distance_matrix[d][i, :])
                connections = np.where(distance_matrix[d][i, :] != 0)[0]

                # Removing an end point does not require reconnecting sensors
                if len(connections) == 2:
                    distance_matrix[d][connections[0], connections[1]] = summed_dist
                    distance_matrix[d][connections[1], connections[0]] = summed_dist

                distance_matrix[d] = np.delete(distance_matrix[d], i, axis=0)
                distance_matrix[d] = np.delete(distance_matrix[d], i, axis=1)

                avg[d] = np.delete(avg[d], i, axis=0)

                # Restart entire process to avoid issue with two neighboring nodes
                i = 0
                sensors = distance_matrix[d].shape[0]
            else:
                i += 1

    # Remove all constant data
    if len(removed_indices[0]) != 0:
        distance_matrix_new = [None]*2
        data_new = [None]*2
        for d in range(2):
            num_sensors = len(data[d])-len(removed_indices[d])

            data_new[d] = np.zeros([num_sensors, 365, 288, 4])
            data_new[d] = np.delete(data[d], removed_indices[d], axis=0)

            sensor_enums_new[d] = remove_items_from_numeric_dict(sensor_enums_new[d], removed_indices[d])

        data = data_new
        sensor_enums = sensor_enums_new

    np.save(save_file + '_parsed_data_north.npy', data[0])
    np.save(save_file + '_parsed_data_south.npy', data[1])

    np.save(save_file + '_distance_matrix_north.npy', distance_matrix[0])
    np.save(save_file + '_distance_matrix_south.npy', distance_matrix[1])

    f = open(save_file + "_sensor_enums.pkl", "wb")
    pickle.dump(sensor_enums, f)
    f.close()


if __name__ == '__main__':
    #read_log_dir('data_prep/i35_5min_bluetoothtraveltimes_2019', 'i35_2019')
    #parse_data('data_prep/i35_2019')
    #remove_constant_sensors('data_prep/i35_2019')
    data, distance_matrix, sensor_enums, time_enums, direction_enums = load_data('data_prep/i35_2019')
    x=21



