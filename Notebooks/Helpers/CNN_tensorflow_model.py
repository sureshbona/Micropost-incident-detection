import keras
from keras import regularizers
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, MaxPool1D, Flatten, Concatenate, Dropout, Conv1D, Bidirectional, GRU, GlobalMaxPooling1D, Dense
import random
import numpy as np
import tensorflow as tf

# Function to create model, required for KerasClassifier
# CNN model taken from https://aclanthology.org/D14-1181/

def CNN_model(seq_len=25, output_size=1, emb_dim=200, drop_prob=0.5,
                 num_filters=100,
                 filter_sizes=[3, 4, 5],
                 dense_layer_activation_fn='sigmoid',
                 seed=42):

    keras.backend.clear_session()
    # random.seed(seed)
    # np.random.seed(seed)
    # tf.random.set_seed(seed)

    # input layer
    text_input_layer = Input(
        shape=(seq_len, emb_dim), dtype='float32', name='input_layer')

    # conv1d + maxpool
    convs_max_f = []

    for filter_size in filter_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            input_shape=(seq_len, emb_dim),
            padding='same',
            activation='relu',
            name='convolution_layer_f_fsize_' + str(filter_size))(text_input_layer)
        
        max_pool = MaxPool1D(pool_size=seq_len, strides=1)(conv)
        convs_max_f.append(max_pool)

    # concatenate conv1d
    cnn_feature_maps_f = Concatenate(
        axis=2, name='concatenation_feature_maps')(convs_max_f)

    # flatten    
    flatten = Flatten()(cnn_feature_maps_f)

    # Dropout
    d2 = Dropout(drop_prob, name='dropout_layer_2')(flatten)

    # FC2
    output_layer = Dense(output_size, activation=dense_layer_activation_fn,
                         name='output_layer')(d2)

    model = Model(inputs=[text_input_layer], outputs=[output_layer])

    # model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

    return model

