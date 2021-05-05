import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np


def num_lines(txt_file):
    file = open(txt_file, "r")
    nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
    line_count = len(nonempty_lines)
    file.close()
    return line_count


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

        if filename.endswith('.xml'):
            tree = ET.parse(filename)
            root = tree.getroot()

            num_sensors = len(root)
            sensors = [None] * num_sensors

            for i in range(num_sensors):
                sensors[i] = root[i].attrib

        else:
            pass

    with open(save_file + '_raw_data' '.pic', 'wb') as filehandle:
        pickle.dump(data, filehandle)
    with open(save_file + '_sensors' '.pic', 'wb') as filehandle:
        pickle.dump(sensors, filehandle)


def load_raw_data(save_file):
    with open(save_file + '_raw_data' '.pic', 'rb') as filehandle:
        raw_data = pickle.load(filehandle)
    with open(save_file + '_sensors' '.pic', 'rb') as filehandle:
        sensors = pickle.load(filehandle)

    return raw_data, sensors


def parse_data(save_file):
    raw_data, sensors = load_raw_data(save_file)

    num_sensors = len(sensors)
    num_days = len(raw_data)
    num_measurements = len(raw_data[0])
    time_segments = 288 # assuming 5 minute time window

    data = np.zeros([2, num_sensors, num_days, time_segments, 5])
    distance_matrix = np.zeros(([2, num_sensors, num_sensors]))

    sensor_enums = {}
    for i in range(len(sensors)):
        sensor_enums[sensors[i]['id']] = i

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
                print('Adding sensor {} to sensor enums'.format(raw_data[day][i][0]))
                sensor_enums[raw_data[day][i][0]] = max(sensor_enums.values()) + 1
                num_sensors += 1

                data_new = np.zeros([2, num_sensors, num_days, time_segments, 5])
                data_new[:, :-1, :, :, :] = data
                data = data_new

                distance_matrix_new = np.zeros([2, num_sensors, num_sensors])
                distance_matrix_new[:, :-1, :-1] = distance_matrix
                distance_matrix = distance_matrix_new

            try:
                sensor_out = sensor_enums[raw_data[day][i][1]]
            except KeyError:
                print('Adding sensor {} to sensor enums'.format(raw_data[day][i][1]))
                sensor_enums[raw_data[day][i][1]] = max(sensor_enums.values()) + 1
                num_sensors += 1

                data_new = np.zeros([2, num_sensors, num_days, time_segments, 5])
                data_new[:, :-1, :, :, :] = data
                data = data_new

                distance_matrix_new = np.zeros([2, num_sensors, num_sensors])
                distance_matrix_new[:, :-1, :-1] = distance_matrix
                distance_matrix = distance_matrix_new

            data[direction, sensor, day, time_stamp, 0] = raw_data[day][i][10]
            data[direction, sensor, day, time_stamp, 1] = raw_data[day][i][11]
            data[direction, sensor, day, time_stamp, 2] = raw_data[day][i][12]
            data[direction, sensor, day, time_stamp, 3] = raw_data[day][i][13]
            data[direction, sensor, day, time_stamp, 4] = raw_data[day][i][14]

            distance_matrix[direction, sensor, sensor_out] = raw_data[day][i][8]


    # Ensure the distance matrix is symmetric
    distance_matrix[0, :, :] = np.maximum(distance_matrix[0, :, :], distance_matrix[0, :, :].transpose())
    distance_matrix[1, :, :] = np.maximum(distance_matrix[1, :, :], distance_matrix[1, :, :].transpose())

    np.save(save_file+'_parsed_data.npy', data)
    np.save(save_file+'_distance_matrix.npy', distance_matrix)

    f = open(save_file+"_sensor_enums.pkl", "wb")
    pickle.dump(sensor_enums, f)
    f.close()

    f = open(save_file+"_time_enums.pkl", "wb")
    pickle.dump(time_enums, f)
    f.close()

    f = open(save_file+"_direction_enums.pkl", "wb")
    pickle.dump(direction_enums, f)
    f.close()


def load_data(save_file):
    with open(save_file + '_parsed_data.npy', 'rb') as f:
        data = np.load(f)
    with open(save_file + '_distance_matrix.npy', 'rb') as f:
        distance_matrix = np.load(f)

    f = open(save_file+"_sensor_enums.pkl", "rb")
    sensor_enums = pickle.load(f)
    f.close()

    f = open(save_file + "_time_enums.pkl", "rb")
    time_enums = pickle.load(f)
    f.close()

    f = open(save_file+"_direction_enums.pkl", "rb")
    direction_enums = pickle.load(f)
    f.close()

    return data, distance_matrix, sensor_enums, time_enums, direction_enums


if __name__ == '__main__':
    #read_log_dir('i35_5min_bluetoothtraveltimes_2019', 'i35_2019')
    #parse_data('data_prep/i35_2019')
    data, distance_matrix, sensor_enums, time_enums, direction_enums = load_data('data_prep/i35_2019')

