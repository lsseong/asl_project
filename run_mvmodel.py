from model import mvmodel
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    n_factor = 2
    seq_length = 5
    n_forward = 2
    seq_model = "linear"

    # when running under window, needs to manually create export/exporter sub folder under this output directory
    out_dir_str = "c:/tf/uob/{}".format(seq_model)

    hparams = {}
    hparams['model'] = seq_model
    hparams['train_data_path'] = "data/train_multi_close_{}_{}_{}.csv".format(n_factor, seq_length, n_forward)
    hparams['eval_data_path'] = "data/train_multi_close_{}_{}_{}.csv".format(n_factor, seq_length, n_forward)
    hparams['n_factor'] = n_factor
    hparams['seq_length'] = seq_length
    hparams['n_forward'] = n_forward
    hparams['learning_rate'] = 0.2
    hparams['train_steps'] = 100
    hparams['batch_size'] = 512
    hparams['eval_delay_secs'] = 10
    hparams['min_eval_frequency'] = 60

    mvmodel.train_and_evaluate(out_dir_str, hparams)
