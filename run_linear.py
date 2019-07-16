sys.path.insert(0, 'model/') #so python can find the utils_ modules

from model import model


if __name__ == '__main__':
    seq_length = 8
    n_forward = 5
    model = "linear"
    out_dir_str = "trained/{}".format(model)

    hparams = {}
    hparams['model'] = model
    hparams['train_data_path'] = "data/train_{}_{}.csv".format(seq_length, n_forward)
    hparams['eval_data_path'] = "data/eval_{}_{}.csv".format(seq_length, n_forward)
    hparams['seq_length'] = 8
    hparams['n_forward'] = 5
    hparams['learning_rate'] = 0.2
    hparams['train_steps'] = 100
    hparams['batch_size'] = 512
    hparams['eval_delay_secs'] = 10
    hparams['min_eval_frequency'] = 60

    model.train_and_evaluate(out_dir_str, hparams)
