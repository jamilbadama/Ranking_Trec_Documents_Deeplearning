
import pandas as pd
FLAGS = None

def load_csv_dataset(data_path):
    train_file  = data_path + '/training_dataset.csv'
    test_file = data_path + '/testing_dataset.csv'
    #Load Training Dataset
    train = pd.read_csv(train_file, encoding='latin1')
    train["Disease"] = train["Disease"].str.lower()
    target_names = train["Disease"].unique()
    train["Disease"] = train["Disease"].astype('category')
    train["Disease"] = train["Disease"].cat.codes
    train = train.sample(frac=1)

    # Load Training Dataset
    test = pd.read_csv(test_file, encoding='latin1')
    test["Disease"] = test["Disease"].str.lower()
    test["Disease"] = test["Disease"].astype('category')
    test["Disease"] = test["Disease"].cat.codes
    test = test.sample(frac=1)

    train_texts = train["Title"] + train["Abstract"]
    train_labels = train["Disease"]
    train["Sentence"] = train_texts
    val_texts = test['Title'] + test['Abstract']
    val_labels = test['Disease']
    test["Sentence"] = val_texts

    return (train_texts, train_labels), (val_texts, val_labels), (train, test), (target_names)



