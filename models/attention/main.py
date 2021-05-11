import self_attention_autoencoder as saa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from data_prep.prep import load_data
from tools.tools import collapse_data, divide_data, split_data, non_overlapping_moving_window_average


def prepare_data(save_file, window_size=1, seq_in_len=512, seq_out_len=1):
    data, distance_matrix, sensor_enums, time_enums, direction_enums = load_data(save_file)
    collapsed_data_north, collapsed_data_south = collapse_data(data)

    train_data_n, val_data_n, test_data_n = divide_data(collapsed_data_north, 8, 1, 1)
    train_data_s, val_data_s, test_data_s = divide_data(collapsed_data_south, 8, 1, 1)

    if window_size != 1:
        new_tr_n = np.zeros([int(train_data_n.shape[0]/window_size), train_data_n.shape[1], train_data_n.shape[2]])
        new_val_n = np.zeros([int(val_data_n.shape[0]/window_size), val_data_n.shape[1], val_data_n.shape[2]])
        new_test_n = np.zeros([int(test_data_n.shape[0]/window_size), test_data_n.shape[1], test_data_n.shape[2]])

        new_tr_s = np.zeros([int(train_data_s.shape[0]/window_size), train_data_s.shape[1], train_data_s.shape[2]])
        new_val_s = np.zeros([int(val_data_s.shape[0]/window_size), val_data_s.shape[1], val_data_s.shape[2]])
        new_test_s = np.zeros([int(test_data_s.shape[0]/window_size), test_data_s.shape[1], test_data_s.shape[2]])

        for i in range(data[0].shape[0]):
            for f in range(data[0].shape[3]):
                new_tr_n[:, i, f] = non_overlapping_moving_window_average(train_data_n[:, i, f], window_size)
                new_val_n[:, i, f] = non_overlapping_moving_window_average(val_data_n[:, i, f], window_size)
                new_test_n[:, i, f] = non_overlapping_moving_window_average(test_data_n[:, i, f], window_size)

        for i in range(data[1].shape[0]):
            for f in range(data[1].shape[3]):
                new_tr_s[:, i, f] = non_overlapping_moving_window_average(train_data_s[:, i, f], window_size)
                new_val_s[:, i, f] = non_overlapping_moving_window_average(val_data_s[:, i, f], window_size)
                new_test_s[:, i, f] = non_overlapping_moving_window_average(test_data_s[:, i, f], window_size)

        train_data_n = new_tr_n
        val_data_n = new_val_n
        test_data_n = new_test_n

        train_data_s = new_tr_s
        val_data_s = new_val_s
        test_data_s = new_test_s


    train_data_in_n, train_data_out_n = split_data(train_data_n, seq_in_len, seq_out_len, 1000, features=[2])
    val_data_in_n, val_data_out_n = split_data(val_data_n, seq_in_len, seq_out_len, 100, features=[2])
    test_data_in_n, test_data_out_n = split_data(test_data_n, seq_in_len, seq_out_len, 100, features=[2])

    train_data_in_s, train_data_out_s = split_data(train_data_s, seq_in_len, seq_out_len, 1000, features=[2])
    val_data_in_s, val_data_out_s = split_data(val_data_s, seq_in_len, seq_out_len, 100, features=[2])
    test_data_in_s, test_data_out_s = split_data(test_data_s, seq_in_len, seq_out_len, 100, features=[2])

    return train_data_in_n, train_data_out_n, val_data_in_n, val_data_out_n, test_data_in_n, test_data_out_n, \
           train_data_in_s, train_data_out_s, val_data_in_s, val_data_out_s, test_data_in_s, test_data_out_s


def train_attn_model(save_file, latent_dim, window_size):
    seq_len = 128
    d_k = 256
    d_v = 256
    n_heads = 12
    ff_dim = 256
    filt_dim = latent_dim + 2

    train_data_in_n, train_data_out_n, val_data_in_n, val_data_out_n, test_data_in_n, test_data_out_n, train_data_in_s,\
    train_data_out_s, val_data_in_s, val_data_out_s, test_data_in_s, test_data_out_s = prepare_data(save_file, window_size=window_size, seq_in_len=seq_len)

    model = saa.create_model(seq_len, d_k, d_v, n_heads, ff_dim, filt_dim, latent_dim)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    callback = tf.keras.callbacks.ModelCheckpoint('models/attention/attn_model_{}.hdf5'.format(window_size),
                                                  monitor='val_loss',
                                                  save_best_only=True, verbose=1)

    model.fit(train_data_in_n, train_data_out_n, batch_size=32, epochs=100, callbacks=[es_callback, callback],
              shuffle=False,
              validation_data=(val_data_in_n, val_data_out_n))

    #model = tf.keras.models.load_model('models/attention/attn_model_{}.hdf5'.format(window_size),
    #                                   custom_objects={'Time2Vector': saa.Time2Vector,
    #                                                   'SingleAttention': saa.SingleAttention,
    #                                                   'MultiAttention': saa.MultiAttention,
    #                                                   'TransformerEncoder': saa.TransformerEncoder})


def poly_attn_predict(save_file, latent_dim):
    data, distance_matrix, sensor_enums, time_enums, direction_enums = load_data(save_file)
    collapsed_data_north, collapsed_data_south = collapse_data(data)
    _, _, test_data_n = divide_data(collapsed_data_north, 8, 1, 1)
    test_data_n = test_data_n[:, :, 2]

    def ready_model(window_size):
        seq_len = 128
        d_k = 256
        d_v = 256
        n_heads = 12
        ff_dim = 256
        filt_dim = latent_dim + 2

        train_data_in_n, train_data_out_n, val_data_in_n, val_data_out_n, test_data_in_n, test_data_out_n, train_data_in_s, \
        train_data_out_s, val_data_in_s, val_data_out_s, test_data_in_s, test_data_out_s = prepare_data(save_file,
                                                                                                        window_size=window_size,
                                                                                                        seq_in_len=seq_len)

        model = saa.create_model(seq_len, d_k, d_v, n_heads, ff_dim, filt_dim, latent_dim)

        model.fit(train_data_in_n, train_data_out_n, batch_size=500, epochs=1, shuffle=False,
                  validation_data=(val_data_in_n, val_data_out_n))

        model = tf.keras.models.load_model('models/attention/attn_model_{}.hdf5'.format(window_size),
                                           custom_objects={'Time2Vector': saa.Time2Vector,
                                                           'SingleAttention': saa.SingleAttention,
                                                           'MultiAttention': saa.MultiAttention,
                                                           'TransformerEncoder': saa.TransformerEncoder})

        return model

    w_model = ready_model(24)
    d_model = ready_model(3)
    m_model = ready_model(1)

    sensors = test_data_n.shape[1]
    n = test_data_n.shape[0]
    w_segments = int(n/24)
    d_segments = int(n/3)

    output_m = np.zeros([n, sensors])
    output_m[:128] = test_data_n[:128]

    output_d = np.zeros([d_segments*128, sensors])
    for i in range(sensors):
        output_d[:128, i] = non_overlapping_moving_window_average(test_data_n[:128*3, i], 3)

    output_w = np.zeros([w_segments * 128, sensors])
    for i in range(sensors):
        output_w[:128, i] = non_overlapping_moving_window_average(test_data_n[:128 * 24, i], 24)

    w_ind = 128 + 1
    d_ind = 128 + 1
    m_ind = 128 + 1

    for i in range(int(n/2)):

        # Prediction
        if i % 24:
            j = int(i/24)
            w_pred = w_model.predict(output_w[tf.newaxis, j:(128 + j), :])
            output_w[(128 + j), :] = 1/3*w_pred
            w_ind = (128 + j) + 1

        if i % 3:
            j = int(i/3)
            d_pred = d_model.predict(output_d[tf.newaxis, j:(128 + j), :])
            output_d[(128 + j), :] = d_pred
            d_ind = (128 + j) + 1

        m_pred = m_model.predict(output_m[tf.newaxis, i:(128 + i), :])
        output_m[(128 + i), :] = m_pred
        m_ind = (128 + i) + 1

        # Fusion
        if i % 24:
            j = int(i/24)
            output_w[(128 + j) + 1, :] += 1 / 3 * np.average(output_d[(d_ind-8):d_ind, :], axis=0)
            output_w[(128 + j) + 1, :] += 1 / 3 * np.average(output_m[(m_ind-24):m_ind, :], axis=0)

        if i % 3:
            j = int(i/3)
            output_d[(128 + j) + 1, :] += 1 / 3 * np.average(output_m[(m_ind-3):m_ind, :], axis=0)
            output_d[(128 + j) + 1, :] += 1 / 3 * output_w[w_ind-1, :]

        output_m[(128 + i), :] += 1 / 3 * output_w[w_ind-1, :]
        output_m[(128 + i), :] += 1 / 3 * output_d[d_ind - 1, :]

    plt.plot(test_data_n[:, 0])
    plt.plot(output_m[:, 0])
    plt.show()


def test_attn(save_file, latent_dim):
    train_data_in_n, train_data_out_n, val_data_in_n, val_data_out_n, test_data_in_n, test_data_out_n, train_data_in_s, \
    train_data_out_s, val_data_in_s, val_data_out_s, test_data_in_s, test_data_out_s = prepare_data(save_file)

    def ready_model():

        seq_len = 128
        d_k = 256
        d_v = 256
        n_heads = 12
        ff_dim = 256
        filt_dim = latent_dim + 2

        model = saa.create_model(seq_len, d_k, d_v, n_heads, ff_dim, filt_dim, latent_dim)

        model.fit(train_data_in_n, train_data_out_n, batch_size=500, epochs=1, shuffle=False,
                  validation_data=(val_data_in_n, val_data_out_n))

        model = tf.keras.models.load_model('attn_model_1_{}_1.hdf5'.format(latent_dim),
                                           custom_objects={'Time2Vector': saa.Time2Vector,
                                                           'SingleAttention': saa.SingleAttention,
                                                           'MultiAttention': saa.MultiAttention,
                                                           'TransformerEncoder': saa.TransformerEncoder})

        return model

    attn_model = ready_model()

    pred_len = 500

    output = np.zeros([2, pred_len+128, latent_dim])
    output[:, :128, :] = test_data_in_n[:2, :128]
    for i in range(pred_len-1):
        prediction_n = attn_model.predict(output[:, i:(i+128), :])
        output[:, 128+i, :] = prediction_n

    plt.plot(test_data_in_n[0, :(128+pred_len), 0])
    plt.plot(output[0, :, 0])





    x = 1



if __name__ == "__main__":
    #train_attn_model('data_prep/i35_2019', 47, 1)
    #train_attn_model('data_prep/i35_2019', 47, 3)
    #train_attn_model('data_prep/i35_2019', 47, 24)
    poly_attn_predict('data_prep/i35_2019', 47)
    #test_attn('data_prep/i35_2019', 47)

