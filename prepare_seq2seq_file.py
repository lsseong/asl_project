"""
To prepare sequence (multivariate inputs) to sequence (label) training file.
Label sequence comes from one of the existing time series

"""
import pandas as pd
import numpy as np
from preprocess import fileutils
from preprocess import dfutils


def shape_data(df, label_column, seq_length, n_forward, sliding_step):
    """
    Slide over time series [N, FEATURES] to produce training features and labels of shape (M, SEQ_LENGTH, FEATURES)
    Each row is a time sequence of various features
    """

    num_features = df.shape[1]

    _df_labels = fileutils.get_column_as(df, label_column, label_column)

    _data = df.to_numpy()
    _output = _df_labels.to_numpy().squeeze()

    batch_size = int((len(_data) - (seq_length + n_forward)) / sliding_step + 1)

    _features = np.zeros((batch_size, seq_length, num_features))
    _label = np.zeros((batch_size, n_forward))

    for i in range(0, batch_size):
        begin_index = i * sliding_step
        stop_index = begin_index + seq_length
        label_stop_index = stop_index + n_forward
        _features[i] = _data[begin_index:stop_index, :]
        _label[i] = _output[stop_index:label_stop_index]

    return _features, _label


def prepare(frames, seq_length, n_forward, sliding_step, filename_prefix, num_decimal=2):
    dfutils.sanity_check(frames)

    num_factor = frames.shape[1]

    features, labels = shape_data(frames, 'UOB', seq_length, n_forward, sliding_step)

    # combined features and label as one file
    num_batches = features.shape[0]

    # reshape features into 2D array
    features = np.reshape(features, (num_batches, -1))

    np.savetxt("data/train_multi_{}_{}_{}_{}.csv".format(filename_prefix, num_factor, seq_length, n_forward),
               np.concatenate((features, labels), axis=1), delimiter=",", fmt="%.{}f".format(num_decimal))


if __name__ == '__main__':
    seq_length = 5
    n_forward = 2
    sliding_step = 1

    dbs_prices = fileutils.read_yahoo_file('yahoo/DBS.csv')
    uob_prices = fileutils.read_yahoo_file('yahoo/UOB.csv')

    # prepare training data using close
    db_close = fileutils.get_column_as(dbs_prices, 'Close', 'DBS')
    uob_close = fileutils.get_column_as(uob_prices, 'Close', 'UOB')
    prepare(dfutils.inner_join(db_close, uob_close), seq_length, n_forward, sliding_step, 'close')

    # prepare training data using adjusted close
    db_adj_close = fileutils.get_column_as(dbs_prices, 'Adj Close', 'DBS')
    uob_adj_close = fileutils.get_column_as(uob_prices, 'Adj Close', 'UOB')
    prepare(dfutils.inner_join(db_adj_close, uob_adj_close), seq_length, n_forward, sliding_step, 'adj_close', num_decimal=6)

    # prepare training data using log returns on adjusted close
    db_returns = fileutils.get_column_as_log_return(dbs_prices, 'Adj Close', 'DBS')
    uob_returns = fileutils.get_column_as_log_return(uob_prices, 'Adj Close', 'UOB')
    prepare(dfutils.inner_join(db_returns, uob_returns), seq_length, n_forward, sliding_step, 'returns', num_decimal=6)
