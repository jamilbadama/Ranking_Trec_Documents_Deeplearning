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

"""The cnn_model layer arragement copied  from @link https://github.com/google/eng-edu/tree/master/ml/guides/text_classification"""
def cnn_model(blocks,
              filters,
              kernel_size,
              embedding_dim,
              dropout_rate,
              pool_size,
              input_shape,
              num_labels,
              num_features):
    op_units, op_activation = get_lastlayer_activation_function(num_labels)

    model = models.Sequential()

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
                 input_shape,
                 num_classes,
                 num_features):

    op_units, op_activation = get_lastlayer_activation_function(num_classes)
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

    op_units, op_activation = get_lastlayer_activation_function(num_classes)
    model = models.Sequential()
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    model.add(Dense(op_units, activation=op_activation))
    return model


def get_lastlayer_activation_function(num_classes):

    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation




