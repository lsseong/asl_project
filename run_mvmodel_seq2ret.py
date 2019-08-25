from model import mvmodel
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    n_factor = 2
    seq_length = 5

    # when running under window, needs to manually create export/exporter sub folder under this output directory
    out_dir_str = "c:/tf/uob/seq2ret"

    hparams = {}
    hparams['model'] = 'seq2seq'
    hparams['train_data_path'] = "data/train_multi_seq2ret_{}_{}_1.csv".format(n_factor, seq_length)
    hparams['eval_data_path'] = "data/train_multi_close_{}_{}_1.csv".format(n_factor, seq_length)
    hparams['n_factor'] = n_factor
    hparams['seq_length'] = seq_length
    hparams['n_forward'] = 1
    hparams['learning_rate'] = 0.2
    hparams['train_steps'] = 10000
    hparams['batch_size'] = 10
    hparams['eval_delay_secs'] = 1
    hparams['min_eval_frequency'] = 10

    mvmodel.train_and_evaluate(out_dir_str, hparams)
