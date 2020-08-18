import bert
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text

topk = 10000


MAX_SEQUENCE_LENGTH = 500

def vectorize_by_sequence(train_texts, val_texts):

    tokenizer = text.Tokenizer(num_words=topk)
    tokenizer.fit_on_texts(train_texts)

    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

def convert_to_tensor(train_texts,val_texts):
    x_train = tf.convert_to_tensor(train_texts, dtype=tf.string)
    x_val = tf.convert_to_tensor(val_texts, dtype=tf.string)
    return x_train, x_val


def bert_tokenizer(bert_layer):
    FullTokenizer = bert.bert_tokenization.FullTokenizer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    return tokenizer



