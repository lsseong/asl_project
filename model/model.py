import tensorflow as tf
import numpy as np

TIME_SERIES_INPUT = "prices"

SEQ_LEN = None


def init(hparams):
    global SEQ_LEN
    SEQ_LEN = hparams['seq_length']


def markov_model(features, n_forward):
    """
    :param features: features tensor of shape [BATCH_SIZE, SEQ_LEN, 1]
    :param n_forward: number of forward predictions
    :return: prediction tensor of shape [BATCH_SIZE, N_FORWARD]
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
    """
    :param features: [BATCH_SIZE, SEQ_LEN]
    :param n_forward:
    :return:
    """
    _Yr = tf.layers.dense(features, n_forward)  # Yr [BATCH_SIZE, N_FORWARD]
    return _Yr


def compute_errors(features, labels, predictions):
    loss = tf.losses.mean_squared_error(labels, predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    mae = tf.metrics.mean_absolute_error(labels, predictions)
    return loss, rmse, mae


def sequence_model(features, labels, mode, params):
    """
    :param features: a dictionary that contains input tensor of shape [BATCH_SIZE, SEQ_LEN]
    :param labels: shape [BATCH_SIZE, N_FORWARD]
    :param model: model name as string
    :return: the EstimatorSpec
    """

    # 1. select the model
    model_functions = {
        'markov': markov_model,
        'linear': linear_model
    }

    model_function = model_functions[params['model']]

    n_forward = params['n_forward']

    predictions = model_function(features[TIME_SERIES_INPUT], n_forward)

    # 2. loss function, training/eval ops
    loss = None
    train_op = None
    eval_metric_ops = None

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # only in training and evaluation mode (not during prediction) that we have labels for computing loss metrics
        loss, rmse, mae = compute_errors(features, labels, predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # this is needed for batch normalization, but has no effect otherwise
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # 2b. set up training operation
                train_op = tf.contrib.layers.optimize_loss(loss,
                                                           tf.train.get_global_step(),
                                                           learning_rate=params['learning_rate'],
                                                           optimizer="Adam")

        # 2c. eval metric
        eval_metric_ops = {
            "RMSE": rmse,
            "MAE": mae
        }

    # 3. Create predictions
    predictions_dict = {"predicted": predictions}

    # 4. return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={
            'predictions': tf.estimator.export.PredictOutput(predictions_dict)}
    )


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
            return {TIME_SERIES_INPUT: features}, labels

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

        # features = [BATCH_SIZE, SEQ_LENGTH]
        # labels = [BATCH_SIZE, N_FORWARD]
        return features, labels

    return _input_fn


# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    feature_placeholders = {
        TIME_SERIES_INPUT: tf.placeholder(tf.float32, [None, SEQ_LEN])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    features[TIME_SERIES_INPUT] = tf.squeeze(features[TIME_SERIES_INPUT], axis=[2])

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


# Create estimator to train and evaluate
def train_and_evaluate(output_dir, hparams):
    init(hparams)

    # ensure file writer cache is clear for TensorBoard events file
    tf.summary.FileWriterCache.clear()

    # parameters
    seq_length = hparams['seq_length']
    batch_size = hparams['batch_size']

    # training data reader
    get_train = read_dataset(hparams['train_data_path'],
                             tf.estimator.ModeKeys.TRAIN,
                             seq_length,
                             hparams['n_forward'],
                             batch_size=batch_size)

    # evaluation data reader
    get_valid = read_dataset(hparams['eval_data_path'],
                             tf.estimator.ModeKeys.EVAL,
                             seq_length,
                             hparams['n_forward'],
                             batch_size=1000)

    # build estimator
    estimator = tf.estimator.Estimator(model_fn=sequence_model,
                                       params=hparams,
                                       config=tf.estimator.RunConfig(save_checkpoints_secs=hparams['min_eval_frequency']),
                                       model_dir=output_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=get_train, max_steps=hparams['train_steps'])

    exporter = tf.estimator.LatestExporter(name="exporter", serving_input_receiver_fn=serving_input_fn, exports_to_keep=None)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_valid,
                                      steps=None,
                                      exporters=exporter,
                                      start_delay_secs=hparams['eval_delay_secs'],
                                      throttle_secs=hparams['min_eval_frequency'])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def test_markov():
    # parameters
    seq_length = 4
    n_forward = 2

    # input and output data
    inputs_ = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    inputs_ = inputs_.reshape((2, 4))
    print(inputs_)

    # placeholder for inputs and output
    features = tf.placeholder(tf.float32, [None, seq_length])  # [BATCH_SIZE, SEQ_LEN]

    predictions = markov_model(features, n_forward)

    # add print operation
    a = tf.print(predictions)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(a, feed_dict={features: inputs_})


def test_read_dataset():
    seq_length = 8
    n_forward = 5

    with tf.Session() as sess:
        fn = read_dataset(filename="../data/train_{}_{}.csv".format(seq_length, n_forward),
                          mode=tf.estimator.ModeKeys.TRAIN,
                          seq_length=seq_length,
                          n_forward=n_forward)

        batch_features, batch_labels = fn()
        features, labels = sess.run([batch_features, batch_labels])
        print("try_out_input_function: features shape = {}".format(features[TIME_SERIES_INPUT].shape))
        print("try_out_input_function: labels shape = {}".format(labels.shape))


def test_train_and_evaluate():
    seq_length = 3
    n_forward = 1
    model = "linear"
    out_dir_str = None

    hparams = {}
    hparams['model'] = model
    hparams['train_data_path'] = "../data/train_{}_{}.csv".format(seq_length, n_forward)
    hparams['eval_data_path'] = "../data/eval_{}_{}.csv".format(seq_length, n_forward)
    hparams['seq_length'] = seq_length
    hparams['n_forward'] = n_forward
    hparams['learning_rate'] = 0.2
    hparams['train_steps'] = 100
    hparams['batch_size'] = 1
    hparams['eval_delay_secs'] = 10
    hparams['min_eval_frequency'] = 60

    train_and_evaluate(out_dir_str, hparams)


if __name__ == '__main__':
    #test_markov()
    #test_read_dataset()
    test_train_and_evaluate()
