"""
To prepare sequence to return training file

"""

import numpy as np
from preprocess import fileutils
from preprocess import dfutils


def shape_data(df, df_labels, seq_length, sliding_step):
    """
    Slide over time series [N, FEATURES] to produce training features (M, SEQ_LENGTH, FEATURES) and labels
    Each row is a time sequence of various features
    """

    num_features = df.shape[1]

    _data = df.to_numpy()
    _output = df_labels.to_numpy().squeeze()

    batch_size = int((len(_data) - seq_length) / sliding_step + 1)

    _features = np.zeros((batch_size, seq_length, num_features))
    _label = np.zeros((batch_size, 1))

    for i in range(0, batch_size):
        begin_index = i * sliding_step
        stop_index = begin_index + seq_length
        _features[i] = _data[begin_index:stop_index, :]
        _label[i] = _output[i]

    return _features, _label


def prepare(features, labels, seq_length, sliding_step, num_decimal=2):
    dfutils.sanity_check(features)

    num_factor = features.shape[1]

    features, labels = shape_data(features, labels, seq_length, sliding_step)

    # combined features and label as one file
    num_batches = features.shape[0]

    # reshape features into 2D array
    features = np.reshape(features, (num_batches, -1))

    # combine features and labels
    data = np.concatenate((features, labels), axis=1)

    # remove nan rows
    data = data[~np.isnan(data).any(axis=1)]

    np.savetxt("data/train_multi_seq2ret_{}_{}_1.csv".format(num_factor, seq_length),
               data,
               delimiter=",",
               fmt="%.{}f".format(num_decimal))


if __name__ == '__main__':
    seq_length = 5
    sliding_step = 1

    dbs_prices = fileutils.read_yahoo_file('yahoo/DBS.csv')
    uob_prices = fileutils.read_yahoo_file('yahoo/UOB.csv')
    df_labels = fileutils.read_yahoo_file('data/UOB_logret_5.csv')

    # get adjusted closes
    db_adj_close = fileutils.get_column_as(dbs_prices, 'Adj Close', 'DBS')
    uob_adj_close = fileutils.get_column_as(uob_prices, 'Adj Close', 'UOB')

    prepare(dfutils.inner_join(db_adj_close, uob_adj_close), df_labels, seq_length, sliding_step, num_decimal=6)
