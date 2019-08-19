import pandas as pd


def sanity_check(df):
    if df.isna().sum().sum() > 0:
        print(df[df.isna().any(axis=1)])
        assert False


def outer_join(df1, df2):
    return pd.concat([df1, df2], axis=1)


def inner_join(df1, df2):
    _df = pd.merge(df1, df2, left_index=True, right_index=True)
    _df = _df.dropna()
    return _df
