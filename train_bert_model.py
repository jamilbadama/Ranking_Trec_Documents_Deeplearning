import argparse
from sklearn import metrics
from sklearn.metrics import classification_report
import build_model
import load_data

MAX_SEQ_LEN = 128
def train_bert_model(data):

    (train_texts, train_labels),(val_texts,val_labels),(train,test),(target_names) = data
    model = build_model.bert_model()
    model = model.fit(train_texts, train_labels)
    # Save model.
    model.save('data/models/bertmodel.model')

    accy = model.score(val_texts, val_labels)

    # make class probability predictions
    y_prob = model.predict_proba(val_texts)
    print("class prob estimates:\n", y_prob)

    # make predictions
    y_pred = model.predict(val_texts)
    print("Accuracy: %0.2f%%" % (metrics.accuracy_score(y_pred, val_labels) * 100))

    print(classification_report(val_labels, y_pred, target_names=target_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/dataset',
                        help='input data directory')
    FLAGS, unparsed = parser.parse_known_args()
    data = load_data.load_csv_dataset(FLAGS.data_dir)
    train_bert_model(data)
