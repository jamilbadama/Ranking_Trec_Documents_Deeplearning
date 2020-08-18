import argparse
import tensorflow as tf
import model_building
import data_loader
import data_vecterization
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
FLAGS = None

topk = 20000

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

    num_labels = len(target_names)

    x_train, x_val, word_index = data_vecterization.vectorize_by_sequence(
        train_texts, val_texts)

    num_features = min(len(word_index) + 1, topk)

    model = model_building.cnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_labels=num_labels,
                                     num_features=num_features)

    if num_labels == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    print("loss", loss)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()

    model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        validation_data=(x_val, val_labels),
        verbose=2,
        batch_size=batch_size)

    # predict probabilities for test set
    yhat_probs = model.predict(x_val, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_val, verbose=0)

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(val_labels, yhat_classes)
    print('Accuracy: %f' % accuracy)
    print(classification_report(val_labels, yhat_classes, target_names=target_names))

    model.save('data/models/cnn_model.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/dataset')
    FLAGS, unparsed = parser.parse_known_args()
    data = data_loader.load_csv_dataset(FLAGS.data_dir)
    train_cnn_model(data)
