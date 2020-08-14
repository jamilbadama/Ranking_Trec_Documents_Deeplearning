import argparse
import tensorflow as tf
import build_model
import explore_data
import load_data
import vectorize_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
FLAGS = None
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

def train_cnn_model(data,
                    learning_rate=1e-3,
                    epochs=14,
                    batch_size=128,
                    blocks=2,
                    filters=64,
                    dropout_rate=0.3,
                    embedding_dim=100,
                    kernel_size=3,
                    pool_size=3):
    (train_texts, train_labels), (val_texts, val_labels), (train, test), (target_names) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = explore_data.get_num_classes(train_labels)

    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
            unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val, word_index = vectorize_data.sequence_vectorize(
        train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    # Create model instance.
    model = build_model.cnn_model(blocks=blocks,
                      filters=filters,
                      kernel_size=kernel_size,
                      embedding_dim=embedding_dim,
                      dropout_rate=dropout_rate,
                      pool_size=pool_size,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes,
                      num_features=num_features)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    print("loss", loss)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    # Train and validate model.
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # predict probabilities for test set
    yhat_probs = model.predict(x_val, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_val, verbose=0)

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(val_labels, yhat_classes)
    print('Accuracy: %f' % accuracy)
    print(classification_report(val_labels, yhat_classes, target_names=target_names))

    # Save model.
    model.save('data/models/cnn_model.model')
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/dataset',
                        help='input data directory')
    FLAGS, unparsed = parser.parse_known_args()
    data = load_data.load_csv_dataset(FLAGS.data_dir)
    train_cnn_model(data)
