"""Module to create model."""

from tensorflow.keras import models
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import SeparableConv1D
from bert_sklearn import BertClassifier

def bert_model():
    model = BertClassifier()
    # model.bert_model = 'bert-base-uncased'
    model.bert_model = 'bert-large-uncased'
    # model.bert_model = 'scibert-basevocab-uncased'
    # model.num_mlp_layers = 10
    model.max_seq_length = 64
    model.epochs = 4
    # model.learning_rate = 4e-5
    model.learning_rate = 2e-5
    model.gradient_accumulation_steps = 1

    return model


def cnn_model(blocks,
              filters,
              kernel_size,
              embedding_dim,
              dropout_rate,
              pool_size,
              input_shape,
              num_classes,
              num_features,
              use_pretrained_embedding=False,
              is_embedding_trainable=False,
              embedding_matrix=None):
    """Creates an instance of a  CNN model """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    print("activiation function", op_activation)
    print("op_units", op_units)
    model = models.Sequential()

    # Add embedding layer. If pre-trained embedding is used add weights to the
    # embeddings layer and set trainable to input is_embedding_trainable flag.
    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable))
    else:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0]))

    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model



def rnn_model(embedding_dim,
                 dropout_rate,
                 input_shape,
                 num_classes,
                 num_features):

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(5, 10)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(op_units, activation=op_activation))
    return model

def biLstm_model(embedding_dim,
                 dropout_rate,
                 input_shape,
                 num_classes,
                 num_features):

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    model.add(Dense(op_units, activation=op_activation))
    return model


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation




