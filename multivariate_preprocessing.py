import pandas as pd
import numpy as np
from datetime import datetime


def read_file(filename):
    to_date = lambda d: datetime.strptime(d, '%Y-%m-%d')
    _df = pd.read_csv(filename, delimiter=",", date_parser=to_date, parse_dates=['Date'], index_col=0)
    return _df


def get_column_as(df, input_name, output_name):
    _df_new = df[[input_name]]
    _df_new.rename(columns={input_name: output_name}, inplace=True)
    return _df_new


def get_column_as_log_return(df, input_name, output_name):
    _df_new = df[[input_name]]
    _df_new[output_name] = np.log(_df_new) - np.log(_df_new.shift(1))

    # first row would be nan
    _df_new.fillna(0, inplace=True)

    del _df_new[input_name]
    return _df_new


def sanity_check(df):
    if df.isna().sum().sum() > 0:
        print(df[df.isna().any(axis=1)])
        assert False


def outer_join(df1, df2):
    return pd.concat([df1, df2], axis=1)


def inner_join(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def shape_data(df, label_column, seq_length, n_forward, sliding_step):
    """
    Slide over time series [N, FEATURES] to produce training features and labels of shape (M, SEQ_LENGTH, FEATURES)
    Each row is a time sequence of various features
    """

    num_features = df.shape[1]

    _df_labels = get_column_as(df, label_column, label_column)

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
    sanity_check(frames)

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

    db_prices = read_file('DB.csv')
    uob_prices = read_file('UOB.csv')

    # prepare training data using close
    db_close = get_column_as(db_prices, 'Close', 'DB')
    uob_close = get_column_as(uob_prices, 'Close', 'UOB')
    prepare(inner_join(db_close, uob_close), seq_length, n_forward, sliding_step, 'close')

    # prepare training data using adjusted close
    db_adj_close = get_column_as(db_prices, 'Adj Close', 'DB')
    uob_adj_close = get_column_as(uob_prices, 'Adj Close', 'UOB')
    prepare(inner_join(db_adj_close, uob_adj_close), seq_length, n_forward, sliding_step, 'adj_close', num_decimal=6)

    # prepare training data using log returns on adjusted close
    db_returns = get_column_as_log_return(db_prices, 'Adj Close', 'DB')
    uob_returns = get_column_as_log_return(uob_prices, 'Adj Close', 'UOB')
    prepare(inner_join(db_returns, uob_returns), seq_length, n_forward, sliding_step, 'returns', num_decimal=6)
