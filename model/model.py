import tensorflow as tf
import numpy as np


def markov_model(features, n_forward):
    """
    :param features: features tensor of shape [BATCH_SIZE, SEQ_LEN, 1]
    :param n_forward: number of forward predictions
    :return: prediction tensor of shape [BATCH_SIZE, 1, N_FORWARD]
    """
    _batch_size = tf.shape(features)[0]
    _seq_length = tf.shape(features)[1]
    _total_size = _batch_size * _seq_length

    # take last value from each sequence
    _last_prices = tf.reshape(features, [_total_size])
    _last_prices = _last_prices[_seq_length - 1::_seq_length]
    _last_prices = tf.reshape(_last_prices, [_batch_size, 1])

    # repeat n_forward number of times
    _prediction = tf.tile(_last_prices, [1, n_forward])
    _prediction = tf.reshape(_prediction, (_batch_size, n_forward))

    return _prediction


def linear_model(features, n_forward):
    pass


def rnn_model(features, labels, mode, params):
    """
    :param features: shape [BATCH_SIZE, SEQ_LEN, 1]
    :param labels: shape [BATCH_SIZE, N_FORWARD]
    :param model: model name as string
    :return: the EstimatorSpec
    """

    # select the model
    model_functions = {
        'markov': markov_model,
        'linear': linear_model
    }

    model_function = model_functions[params['model']]

    n_forward = tf.shape(labels)[1]

    Yout_ = model_function(features, n_forward)

    # loss function
    #loss = tf.losses.mean_squared_error(Yout_, labels)

    # optimisation operation

    return Yout_


# read data and convert to needed format
def read_dataset(filename, mode, seq_length, n_forward, batch_size=512):
    def _input_fn():
        # use 0.0 to indicate float type
        record_defaults = [0.0] * (seq_length + n_forward)

        def decode_csv(row):
            # row is a string tensor containing the contents of one row
            features_list = tf.decode_csv(row, record_defaults=record_defaults)

            features = features_list[:seq_length]
            labels = features_list[seq_length:]

            features = tf.stack(features)
            labels = tf.stack(labels)
            return {"prices": features}, labels

        # Create list of file names
        dataset = tf.data.Dataset.list_files(filename)

        # Read in data from files
        dataset = dataset.flat_map(tf.data.TextLineDataset)

        # Parse text lines as comma-separated values (CSV)
        dataset = dataset.map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # loop indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn


def sequence_regressor():
    pass


def train_and_evaluate(output_dir, hparams):
    # ensure file writer cache is clear for TensorBoard events file
    tf.summary.FileWriterCache.clear()

    # training data reader
    get_train = read_dataset(hparams['train_data_path'],
                             tf.estimator.ModeKeys.TRAIN,
                             hparams['seq_length'],
                             hparams['n_forward'])

    # evaluation data reader
    #get_valid = read_dataset(hparams['eval_data_path'], tf.estimator.ModeKeys.EVAL, 1000)

    estimator = None
    train_spec = None
    eval_spec = None
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def test_markov():
    # parameters
    seq_length = 4
    n_forward = 2
    mode = tf.estimator.ModeKeys.TRAIN

    params = {}
    params['model'] = "markov"

    # input and output data
    inputs_ = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    inputs_ = inputs_.reshape((2, 4, 1))
    print(inputs_)

    outputs_ = np.array([100, 101, 102, 103])
    outputs_ = outputs_.reshape((2, 2, 1))
    print(outputs_)

    # placeholder for inputs and output
    features = tf.placeholder(tf.float32, [None, seq_length, 1])  # [BATCH_SIZE, SEQ_LEN, 1]
    labels = tf.placeholder(tf.float32, [None, n_forward, 1])  # [BATCH_SIZE, N_FORWARD, 1]

    predictions = rnn_model(features, labels, mode, params)

    # add print operation
    a = tf.print(predictions)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(a, feed_dict={features: inputs_, labels: outputs_})


def test_read_dataset():
    seq_length = 8
    n_forward = 5

    with tf.Session() as sess:
        fn = read_dataset(filename="../train_{}_{}.csv".format(seq_length, n_forward),
                          mode=tf.estimator.ModeKeys.TRAIN,
                          seq_length=seq_length,
                          n_forward=n_forward)

        batch_features, batch_labels = fn()
        features, labels = sess.run([batch_features, batch_labels])
        print("try_out_input_function: features shape = {}".format(features['prices'].shape))
        print("try_out_input_function: labels shape = {}".format(labels.shape))

def test_train_and_evaluate():
    seq_length = 8
    n_forward = 5
    model = "markov"
    out_dir_str = "../trained/{}".format(model)

    hparams = {}
    hparams['model'] = model
    hparams['train_data_path'] = "../train_{}_{}.csv".format(seq_length, n_forward)
    hparams['eval_data_path'] = "../eval_{}_{}.csv".format(seq_length, n_forward)
    hparams['seq_length'] = 8
    hparams['n_forward'] = 5

    train_and_evaluate(out_dir_str, hparams)


if __name__ == '__main__':
    test_read_dataset()
