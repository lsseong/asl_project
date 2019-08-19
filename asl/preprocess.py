import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split


def read_file(filename):
    to_datetime = lambda d: datetime.strptime(d, '%d/%m/%Y %H:%M')
    _df = pd.read_csv(filename, delimiter=";", date_parser=to_datetime, parse_dates=['date (UTC)'], index_col=0)
    return _df


def shape_data(df, seq_length, n_forward, sliding_step):
    """
    slide over time series [N] to produce training features and labels
    :param df: time series as dataframe
    :param seq_length:
    :param n_forward:
    :param sliding_step:
    :return:
        features: [BATCH_SIZE, SEQ_LENGTH]
        labels: [BATCH_SIZE, N_FORWARD]
    """
    _data = df.to_numpy().squeeze()

    batch_size = int((len(_data) - (seq_length+n_forward)) / sliding_step + 1)

    _features = np.zeros((batch_size, seq_length))
    _label = np.zeros((batch_size, n_forward))

    for i in range(0, batch_size):
        begin_index = i * sliding_step
        stop_index = begin_index + seq_length
        label_stop_index = stop_index + n_forward
        _features[i] = _data[begin_index:stop_index]
        _label[i] = _data[stop_index:label_stop_index]

    return _features, _label


def process(filename, seq_length, n_forward):
    df = read_file(filename)
    features, labels = shape_data(df, seq_length, n_forward, sliding_step=1)

    features_train, features_eval, labels_train, labels_eval = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.3,
                                                                                random_state=42)

    # combined features and label as one file
    np.savetxt("data/train_{}_{}.csv".format(seq_length, n_forward),
               np.concatenate((features_train, labels_train), axis=1), delimiter=",", fmt="%.2f")

    np.savetxt("data/eval_{}_{}.csv".format(seq_length, n_forward),
               np.concatenate((features_eval, labels_eval), axis=1), delimiter=",", fmt="%.2f")


if __name__ == '__main__':
    process('price.csv', seq_length=8, n_forward=5)

