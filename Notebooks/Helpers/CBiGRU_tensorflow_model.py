import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout, Conv1D, Bidirectional, GRU, GlobalMaxPooling1D, Dense
import random
import numpy as np
import tensorflow as tf

# Function to create model, required for KerasClassifier
# CNN with BiGRU model taken from https://aclanthology.org/S19-2127/

def CBiGRU_model(input_size=150, output_size=1, emb_dim=400, drop_prob=0.2,
                 num_filters=128,
                 filter_sizes=[2, 3, 4, 5],
                 dense_layer_activation_fn='sigmoid',
                 seed=42):

    keras.backend.clear_session()
    #random.seed(seed)
    #np.random.seed(seed)
    #tf.random.set_seed(seed)

    # input layer
    text_input_layer = Input(
        shape=(input_size, emb_dim), dtype='float32', name='input_layer')

    # dropout
    d0 = Dropout(drop_prob, name='dropout_layer_embedding')(text_input_layer)

    # conv1d
    convs_f = []

    for filter_size in filter_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            input_shape=(input_size, emb_dim),
            padding='same',
            activation='relu',
            name='convolution_layer_f_fsize_' + str(filter_size))(d0)
        convs_f.append(conv)

    # concatenate conv1d
    cnn_feature_maps_f = keras.layers.Concatenate(
        axis=2, name='concatenation_feature_maps')(convs_f)

    # BiGRU
    sentence_encoder = Bidirectional(
        GRU(64, return_sequences=True),
        name='bidirectional_gru')(cnn_feature_maps_f)

    # GlobalMaxPooling1D
    text_gmaxpool = GlobalMaxPooling1D(
        name='gmax_pooling_layer')(sentence_encoder)

    # FC 1
    fc_layer1 = Dense(
        32, activation="relu", name='hidden_layer')(text_gmaxpool)

    # Dropout
    d2 = Dropout(drop_prob, name='dropout_layer_2')(fc_layer1)

    # FC2
    output_layer = Dense(output_size, activation=dense_layer_activation_fn,
                         name='output_layer')(d2)

    model = Model(inputs=[text_input_layer], outputs=[output_layer])

    # model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
    return model
