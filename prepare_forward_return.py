from preprocess import fileutils


if __name__ == '__main__':
    forward_step = 5
    uob_prices = fileutils.read_yahoo_file('yahoo/UOB.csv')
    df = fileutils.get_column_as_forward_log_return(uob_prices, 'Adj Close', 'Value', 5)
    df.to_csv('data/UOB_logret_{}.csv'.format(forward_step))



