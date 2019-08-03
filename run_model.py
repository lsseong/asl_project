from model import model


if __name__ == '__main__':
    seq_length = 8
    n_forward = 5
    seq_model = "cnn"

    # when running under window, needs to manually create export/exporter sub folder under this output directory
    out_dir_str = "c:/tf/trained/{}".format(seq_model)

    hparams = {}
    hparams['model'] = seq_model
    hparams['train_data_path'] = "data/train_{}_{}.csv".format(seq_length, n_forward)
    hparams['eval_data_path'] = "data/eval_{}_{}.csv".format(seq_length, n_forward)
    hparams['seq_length'] = seq_length
    hparams['n_forward'] = n_forward
    hparams['n_factor'] = 1
    hparams['learning_rate'] = 0.2
    hparams['train_steps'] = 100
    hparams['batch_size'] = 512
    hparams['eval_delay_secs'] = 10
    hparams['min_eval_frequency'] = 60

    model.train_and_evaluate(out_dir_str, hparams)
