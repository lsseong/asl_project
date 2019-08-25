import pandas as pd
import numpy as np
from datetime import datetime


def read_yahoo_file(filename):
    return read_csv(filename, '%Y-%m-%d')


def read_csv(filename, date_pattern):
    to_date = lambda d: datetime.strptime(d, date_pattern)
    _df = pd.read_csv(filename, delimiter=",", date_parser=to_date, parse_dates=['Date'], index_col=0)
    return _df


def get_column_as(df, input_name, output_name):
    _df_new = df[[input_name]]
    _df_new.rename(columns={input_name: output_name}, inplace=True)
    return _df_new


def get_column_as_log_return(df, input_name, output_name, distance=1):
    _df_new = df[[input_name]]
    _df_new[output_name] = np.log(_df_new) - np.log(_df_new.shift(distance))

    # first few rows (equals distance) would be nan
    _df_new.fillna(0, inplace=True)

    del _df_new[input_name]
    return _df_new


def get_column_as_forward_log_return(df, input_name, output_name, distance=1):
    _df_new = df[[input_name]]
    _df_log = np.log(_df_new)
    _df_new[output_name] = _df_log.shift(-distance) - np.log(_df_new)

    del _df_new[input_name]
    return _df_new
