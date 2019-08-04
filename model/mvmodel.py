"""
This module implements multivariate (multiple factors) sequence to sequence prediction models.

"""
import tensorflow as tf

TIME_SERIES_INPUT = "prices"
GO_TOKEN = -1.0
N_FACTOR = None
SEQ_LEN = None


def linear_model(features, n_forward):
    """
    :param features: [BATCH_SIZE, SEQ_LEN, N_FACTOR]
    :param n_forward: number of sequence to predict
    :return:
    """

    _X = tf.reshape(features, [-1, SEQ_LEN * N_FACTOR])
    _Y = tf.layers.dense(_X, n_forward, name='output')  # Y [BATCH_SIZE, N_FORWARD]
    return _Y


def init(hparams):
    # update global variable inside init function
    global SEQ_LEN, N_FACTOR
    N_FACTOR = hparams['n_factor']
    SEQ_LEN = hparams['seq_length']


# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    feature_placeholders = {
        TIME_SERIES_INPUT: tf.placeholder(tf.float32, [None, SEQ_LEN, N_FACTOR])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


# read data and convert to needed format
def read_dataset(filename, mode, n_factor, seq_length, n_forward, batch_size):
    def _input_fn():
        # use 0.0 to indicate float type
        num_features = n_factor * seq_length
        record_defaults = [0.0] * (num_features + n_forward)

        def decode_csv(row):
            # row is a string tensor containing the contents of one row
            features_list = tf.decode_csv(row, record_defaults=record_defaults)

            # getting a list of tensors
            _features = features_list[:num_features]
            _features = tf.reshape(_features, [seq_length, n_factor])

            _labels = features_list[num_features:]
            _labels = tf.stack(_labels)

            return {TIME_SERIES_INPUT: _features}, _labels

        # Create list of file names
        dataset = tf.data.Dataset.list_files(filename)

        # Read in data from files
        dataset = dataset.flat_map(tf.data.TextLineDataset)

        # Parse text lines as comma-separated values (CSV)
        dataset = dataset.map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # loop indefinitely
            dataset = dataset.shuffle(buffer_size=10*batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        features, labels = dataset.make_one_shot_iterator().get_next()

        # features = [BATCH_SIZE, SEQ_LENGTH, N_FACTOR]
        # labels = [BATCH_SIZE, N_FORWARD]
        return features, labels

    return _input_fn


def compute_errors(labels, predictions):
    loss = tf.losses.mean_squared_error(labels, predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    mae = tf.metrics.mean_absolute_error(labels, predictions)
    return loss, rmse, mae


def build_basic_model(features, labels, mode, params):
    # 1. select the model
    model_functions = {
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
        loss, rmse, mae = compute_errors(labels, predictions)

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

    return predictions, loss, train_op, eval_metric_ops


def build_seq2seq_model(features, labels, mode, params):
    pass


def build_model(features, labels, mode, params):
    """
    :param features: a dictionary that contains input tensor of shape [BATCH_SIZE, SEQ_LEN]
    :param labels: shape [BATCH_SIZE, N_FORWARD]
    :param model: model name as string
    :return: the EstimatorSpec
    """

    _model = params['model']
    if _model == "seq2seq":
        predictions, loss, train_op, eval_metric_ops = build_seq2seq_model(features, labels, mode, params)
    else:
        predictions, loss, train_op, eval_metric_ops = build_basic_model(features, labels, mode, params)

    # put predictions in an output dictionary
    predictions_dict = {"predicted": predictions}

    # return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={
            'predictions': tf.estimator.export.PredictOutput(predictions_dict)}
    )


# Create estimator to train and evaluate
def train_and_evaluate(output_dir, hparams):
    init(hparams)

    # ensure file writer cache is clear for TensorBoard events file
    tf.summary.FileWriterCache.clear()

    # parameters
    n_factor = hparams['n_factor']
    seq_length = hparams['seq_length']
    batch_size = hparams['batch_size']
    n_forward = hparams['n_forward']

    # training data reader
    get_train_dataset = read_dataset(hparams['train_data_path'],
                                     tf.estimator.ModeKeys.TRAIN,
                                     n_factor,
                                     seq_length,
                                     n_forward,
                                     batch_size=batch_size)

    # evaluation data reader
    get_valid_dataset = read_dataset(hparams['eval_data_path'],
                                     tf.estimator.ModeKeys.EVAL,
                                     n_factor,
                                     seq_length,
                                     n_forward,
                                     batch_size=batch_size)

    # build estimator
    estimator = tf.estimator.Estimator(model_fn=build_model,
                                       params=hparams,
                                       config=tf.estimator.RunConfig(save_checkpoints_secs=hparams['min_eval_frequency']),
                                       model_dir=output_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=get_train_dataset, max_steps=hparams['train_steps'])

    exporter = tf.estimator.LatestExporter(name="exporter", serving_input_receiver_fn=serving_input_fn, exports_to_keep=None)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_valid_dataset,
                                      steps=None,
                                      exporters=exporter,
                                      start_delay_secs=hparams['eval_delay_secs'],
                                      throttle_secs=hparams['min_eval_frequency'])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

