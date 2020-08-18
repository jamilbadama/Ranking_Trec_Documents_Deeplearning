import argparse
import tensorflow as tf
import model_building
import data_loader
import data_vecterization

FLAGS = None
topk = 20000
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def train_bilstm_model(data,
                       learning_rate=1e-3,
                       epochs=14,
                       batch_size=128,
                       dropout_rate=0.2,
                       embedding_dim=100):
    (train_texts, train_labels), (val_texts, val_labels), (train, test), (target_names) = data

    num_labels = len(target_names)

    unexpected_labels = [v for v in val_labels if v not in range(num_labels)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
            unexpected_labels=unexpected_labels))
    x_train, x_val, word_index = data_vecterization.vectorize_by_sequence(
        train_texts, val_texts)

    num_features = min(len(word_index) + 1, topk)

    model = model_building.biLstm_model(embedding_dim=embedding_dim,
                                        dropout_rate=dropout_rate,
                                        input_shape=x_train.shape[1:],
                                        num_classes=num_labels,
                                        num_features=num_features)

    if num_labels == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    model.fit(
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

    model.save('data/models/bilstm_model.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/dataset')
    FLAGS, unparsed = parser.parse_known_args()
    data = data_loader.load_csv_dataset(FLAGS.data_dir)
    train_bilstm_model(data)
