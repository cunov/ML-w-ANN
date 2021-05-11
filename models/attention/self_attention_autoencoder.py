from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

Layer = keras.layers.Layer


class Time2Vector(Layer):  # Time embedding layer
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:, :, :], axis=-1)  # Convert (batch, seq_len, 5) to (batch, seq_len)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)  # (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # (batch, seq_len, 2)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


class SingleAttention(Layer):  # Attention layer
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = layers.Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')
        self.key = layers.Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')
        self.value = layers.Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out


class MultiAttention(Layer):  # Multihead attention
    def __init__(self, d_k, d_v, n_heads, filt_dim):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.filt_dim = filt_dim
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
        self.linear = layers.Dense(self.filt_dim, input_shape=input_shape, kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear


class TransformerEncoder(Layer):  # Combining everything into a Transformer encoder
    def __init__(self, d_k, d_v, n_heads, ff_dim, filt_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.filt_dim = filt_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads, self.filt_dim)
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        self.attn_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        self.ff_conv1D_2 = layers.Conv1D(filters=self.filt_dim,
                                  kernel_size=1)  # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        #self.ff_dropout = layers.Dropout(self.dropout_rate)
        self.ff_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'filt_dim': self.filt_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config


def create_model(seq_len, d_k, d_v, n_heads, ff_dim, filt_dim, latent_dim):
  '''Initialize time and transformer layers'''
  time_embedding = Time2Vector(seq_len)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, filt_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, filt_dim)
  attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, filt_dim)
  attn_layer4 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, filt_dim)
  attn_layer5 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, filt_dim)
  attn_layer6 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, filt_dim)
  attn_layer7 = TransformerEncoder(d_k, d_v, n_heads, ff_dim, filt_dim)

  '''Construct model'''
  in_seq = layers.Input(shape=(seq_len, latent_dim))
  x = time_embedding(in_seq)
  x = layers.Concatenate(axis=-1)([in_seq, x])
  x = attn_layer1((x, x, x))
  #x = attn_layer2((x, x, x))
  #x = attn_layer3((x, x, x))

  #x = layers.Flatten()(x)
  #x = layers.Dropout(0.1)(x)
  #x = layers.Dense(latent_dim, activation='relu')(x)
  #x = layers.Dropout(0.1)(x)
  x = x[:, seq_len-1, :]
  out = layers.Dense(latent_dim, activation='linear')(x)

  model = Model(inputs=in_seq, outputs=out)
  model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
  return model


