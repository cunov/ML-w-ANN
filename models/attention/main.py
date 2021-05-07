import self_attention_autoencoder as saa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from data_prep.prep import load_data
from tools.tools import collapse_data, divide_data, split_data


def prepare_data(save_file, seq_in_len=128, seq_out_len=1):
    data, distance_matrix, sensor_enums, time_enums, direction_enums = load_data(save_file)
    collapsed_data_north, collapsed_data_south = collapse_data(data)

    train_data_n, val_data_n, test_data_n = divide_data(collapsed_data_north, 8, 1, 1)
    train_data_s, val_data_s, test_data_s = divide_data(collapsed_data_south, 8, 1, 1)

    train_data_in_n, train_data_out_n = split_data(train_data_n, seq_in_len, seq_out_len, 1000, features=[2])
    val_data_in_n, val_data_out_n = split_data(val_data_n, seq_in_len, seq_out_len, 100, features=[2])
    test_data_in_n, test_data_out_n = split_data(test_data_n, seq_in_len*10, seq_out_len, 100, features=[2])

    train_data_in_s, train_data_out_s = split_data(train_data_s, seq_in_len, seq_out_len, 1000, features=[2])
    val_data_in_s, val_data_out_s = split_data(val_data_s, seq_in_len, seq_out_len, 100, features=[2])
    test_data_in_s, test_data_out_s = split_data(test_data_s, seq_in_len*10, seq_out_len, 100, features=[2])

    return train_data_in_n, train_data_out_n, val_data_in_n, val_data_out_n, test_data_in_n, test_data_out_n, \
           train_data_in_s, train_data_out_s, val_data_in_s, val_data_out_s, test_data_in_s, test_data_out_s


def train_attn_model_1(save_file, latent_dim):
    seq_len = 128
    d_k = 256
    d_v = 256
    n_heads = 12
    ff_dim = 256
    filt_dim = latent_dim + 2

    train_data_in_n, train_data_out_n, val_data_in_n, val_data_out_n, test_data_in_n, test_data_out_n, train_data_in_s,\
    train_data_out_s, val_data_in_s, val_data_out_s, test_data_in_s, test_data_out_s = prepare_data(save_file)

    model = saa.create_model(seq_len, d_k, d_v, n_heads, ff_dim, filt_dim, latent_dim)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    callback = tf.keras.callbacks.ModelCheckpoint('attn_model_1_{}_1.hdf5'.format(latent_dim),
                                                  monitor='val_loss',
                                                  save_best_only=True, verbose=1)

    model.fit(train_data_in_n, train_data_out_n, batch_size=32, epochs=100, callbacks=[es_callback, callback],
              shuffle=False,
              validation_data=(val_data_in_n, val_data_out_n))

    model = tf.keras.models.load_model('attn_model_1_{}_1.hdf5'.format(latent_dim),
                                       custom_objects={'Time2Vector': saa.Time2Vector,
                                                       'SingleAttention': saa.SingleAttention,
                                                       'MultiAttention': saa.MultiAttention,
                                                       'TransformerEncoder': saa.TransformerEncoder})


    en_test_prediction = model.predict(test_data_in_n)


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
    #train_attn_model_1('data_prep/i35_2019', 47)
    test_attn('data_prep/i35_2019', 47)

