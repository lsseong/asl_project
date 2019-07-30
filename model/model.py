import tensorflow as tf
from tensorflow.python.framework import dtypes

TIME_SERIES_INPUT = "prices"
GO_TOKEN = -1.0
SEQ_LEN = None


def init(hparams):
    global SEQ_LEN
    SEQ_LEN = hparams['seq_length']


def linear_model(features, n_forward):
    """
    :param features: [BATCH_SIZE, SEQ_LEN]
    :param n_forward:
    :param mode: running mode (training, validation or inference)
    :return:
    """
    _Yr = tf.layers.dense(features, n_forward)  # Yr [BATCH_SIZE, N_FORWARD]
    return _Yr


def dnn_model(features, n_forward):
    h1 = tf.layers.dense(features, SEQ_LEN * 2)     # [BATCH_SIZE, SEQ_LENGTH * 2]
    h2 = tf.layers.dense(h1, SEQ_LEN)               # [BATCH_SIZE, SEQ_LENGTH]
    _Yr = tf.layers.dense(h2, n_forward)            # Yr [BATCH_SIZE, N_FORWARD]
    return _Yr


def cnn_model(features, n_forward):
    X = tf.reshape(features, [-1, SEQ_LEN, 1])  # as a 1D "sequence" with only one time-series observation (height)

    c1 = tf.layers.conv1d(X,
                          filters=SEQ_LEN // 2,
                          kernel_size=3,
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu)

    p1 = tf.layers.max_pooling1d(c1, pool_size=2, strides=2)

    c2 = tf.layers.conv1d(p1,
                          filters=SEQ_LEN // 2,
                          kernel_size=3,
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu)

    p2 = tf.layers.max_pooling1d(c2, pool_size=2, strides=2)

    outlen = p2.shape[1] * p2.shape[2]
    c2flat = tf.reshape(p2, [-1, outlen])
    predictions = tf.layers.dense(c2flat, n_forward, activation=None)  # linear output: regression

    return predictions


def rnn_model(features, n_forward):
    cell_size = SEQ_LEN  # size of the internal state in each of the cells

    # 1. dynamic_rnn needs 3D shape: [BATCH_SIZE, SEQ_LEN, 1]
    x = tf.reshape(features, [-1, SEQ_LEN, 1])

    # 2. configure the RNN
    cell = tf.nn.rnn_cell.GRUCell(cell_size)

    # https://stats.stackexchange.com/questions/330176/what-is-the-output-of-a-tf-nn-dynamic-rnn
    outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # 3. pass rnn output through a dense layer
    h1 = tf.layers.dense(state, SEQ_LEN // 2, activation=tf.nn.relu)
    predictions = tf.layers.dense(h1, n_forward, activation=None)  # (?, N_FORWARD)
    return predictions


def lstm_model(features, n_forward):
    CELL_SIZE = SEQ_LEN # size of the internal state in each of the cells

    # 1. dynamic_rnn needs 3D shape: [BATCH_SIZE, N_INPUTS, 1]
    x = tf.reshape(features, [-1, SEQ_LEN, 1])

    # 2. configure the LSTM
    cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE)
    outputs, (cell_state, hidden_state) = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # 3. pass rnn output through a dense layer
    h1 = tf.layers.dense(hidden_state, SEQ_LEN // 2, activation=tf.nn.relu)
    predictions = tf.layers.dense(h1, n_forward, activation=None)  # (?, 1)
    return predictions


def seq2seq_model(features, labels, n_forward):

    # dynamic_rnn needs 3D shape: [BATCH_SIZE, SEQ_LEN, 1]
    x = tf.reshape(features, [-1, SEQ_LEN, 1])

    # encoding layer
    encoding_dimension = SEQ_LEN // 2
    encoder_outputs, encoder_state = encoding_layer(x, encoding_dimension)

    _batch_size = tf.shape(features)[0]

    # decoding layer
    decoding_dimension = encoding_dimension
    train_output, infer_output = decoding_layer(labels,
                                                encoder_state,
                                                n_forward,
                                                decoding_dimension,
                                                _batch_size)

    return train_output, infer_output


def encoding_layer(features, encoding_dimension):
    """
    :param features: input features [BATCH_SIZE, SEQ_LENGTH, 1]
    :param encoding_dimension:
    :param keep_prob:
    :return:
        outputs [BATCH_SIZE, SEQ_LENGTH, ENCODE_DIMENSION]
        state [BATCH_SIZE, ENCODE_DIMENSION]
    """
    encoder_cell = tf.nn.rnn_cell.GRUCell(encoding_dimension, name="encoder_cell")
    wrapped_encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=0.9)
    outputs, state = tf.nn.dynamic_rnn(wrapped_encoder_cell, features, dtype=tf.float32)
    return outputs, state


def process_decoder_input(target_data, batch_size):
    go_id = GO_TOKEN

    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat


def decoding_layer(decoder_inputs, encoder_state, n_forward, decoding_dimension, batch_size):
    """
    :param decoder_inputs: [BATCH_SIZE, N_FORWARD]
    :param encoder_state [BATCH_SIZE, ENCODE_DIMENSION]
    :return: train_output
    """

    # RNN cells for decoding
    cell = tf.nn.rnn_cell.GRUCell(decoding_dimension, name="decoder_cell")

    """
    Decoding model can be thought of two separate processes, training and inference.   
    The helper instance is the one that differs in training and inference. 
    During training, we want the inputs to be fed to the decoder, while during inference, 
    we want the output of the decoder in time-step (t) to be passed as the input to the decoder in time step (t+1).
    
    Reference: 
        https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f
        https://stackoverflow.com/questions/43622778/tensorflow-sequence-to-sequence-model-using-the-seq2seq-api-ver-1-1-and-above
    """

    # prediction is a float (i.e. the price) at each time step
    prediction_layer = tf.layers.Dense(units=1)

    train_output = None
    with tf.variable_scope("decode"):
        # During PREDICT mode, the decoder_inputs (i.e. labels) is none so we can't have a training model
        if decoder_inputs is not None:
            train_output = decoding_layer_train(decoder_inputs,
                                                encoder_state,
                                                cell,
                                                n_forward,
                                                prediction_layer,
                                                batch_size)

    # Share weights with training decoder
    with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
        infer_output = decoding_layer_infer(encoder_state, cell, n_forward, prediction_layer, batch_size)

    return train_output, infer_output


def decoding_layer_train(decoder_inputs, encoder_state, cell, n_forward, prediction_layer, batch_size):
    """
    :param decoder_inputs: [BATCH_SIZE, N_FORWARD]
    :param encoder_state: [BATCH_SIZE, ENCODE_DIMENSION]
    :param cell:
    :param n_forward:
    :param prediction_layer:
    :param batch_size
    :return:
    """
    # wrap with a dropout layer
    decoder_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.9)

    # append a column of GO_TOKEN
    go_tokens = tf.fill(tf.shape(decoder_inputs), GO_TOKEN)
    go_tokens = go_tokens[:, 0]
    go_tokens = tf.reshape(go_tokens, [-1, 1])
    expanded_decoder_inputs = tf.concat([go_tokens, decoder_inputs], axis=1)

    # decoder inputs [BATCH_SIZE, N_FORWARD+1, 1]
    expanded_decoder_inputs = tf.reshape(expanded_decoder_inputs, [-1, n_forward+1, 1])

    # sequence length, ensure same batch length as decoder inputs
    sequence_length = tf.fill(tf.shape(decoder_inputs), n_forward)[:, 0]

    # Helper for the training process
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=expanded_decoder_inputs,
                                                        sequence_length=sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                              training_helper,
                                              initial_state=encoder_state,
                                              output_layer=prediction_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=n_forward)

    return outputs


def decoding_layer_infer(encoder_state, cell, n_forward, prediction_layer, batch_size):
    """
    :param encoder_state: [BATCH_SIZE, ENCODE_DIMENSION]
    :param cell:
    :param n_forward:
    :param prediction_layer:
    :param batch_size:
    :return:
    """

    decoder_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.9)

    start_tokens = tf.tile(tf.constant([GO_TOKEN], dtype=tf.float32), [batch_size], name='start_tokens')
    start_tokens = tf.reshape(start_tokens, [-1, 1])

    # This is an inference helper without embedding. The sample_ids are the
    # actual output in this case (not dealing with any logits here).
    # The end_fn is always False because the data is provided by a generator
    # that will stop once it reaches output_size. This could be
    # extended to outputs of various size if we append end tokens, and have
    # the end_fn check if sample_id return True for an end token.
    inference_helper = tf.contrib.seq2seq.InferenceHelper(
        sample_fn=lambda outputs: outputs,
        sample_shape=[1],
        sample_dtype=dtypes.float32,
        start_inputs=start_tokens,
        end_fn=lambda sample_ids: False)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                        inference_helper,
                                                        encoder_state,
                                                        prediction_layer)

    inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=n_forward)

    return inference_decoder_output


def compute_errors(labels, predictions):
    loss = tf.losses.mean_squared_error(labels, predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    mae = tf.metrics.mean_absolute_error(labels, predictions)
    return loss, rmse, mae


def build_basic_model(features, labels, mode, params):
    # 1. select the model
    model_functions = {
        'linear': linear_model,
        'dnn': dnn_model,
        'cnn': cnn_model,
        'rnn': rnn_model,
        'lstm': lstm_model
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
    predictions = None # [BATCH_SIZE, SEQ_LENGTH, 1]
    loss = None
    train_op = None
    eval_metric_ops = None

    n_forward = params['n_forward']

    training_decoder_output, inference_decoder_output = seq2seq_model(features[TIME_SERIES_INPUT], labels, n_forward)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # Predict using training output
        predictions = training_decoder_output.rnn_output    # [BATCH_SIZE, ?, 1]
        predictions = tf.squeeze(predictions)               # [BATCH_SIZE, ?]
        loss, rmse, mae = compute_errors(labels, predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

            # Apply gradient clipper.
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients, global_step=tf.train.get_global_step())

        eval_metric_ops = {
            "RMSE": rmse,
            "MAE": mae
        }
    elif mode == tf.estimator.ModeKeys.PREDICT:
        # Predict using inference output
        predictions = inference_decoder_output.rnn_output

    return predictions, loss, train_op, eval_metric_ops


def sequence_model(features, labels, mode, params):
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


# read data and convert to needed format
def read_dataset(filename, mode, seq_length, n_forward, batch_size):
    def _input_fn():
        # use 0.0 to indicate float type
        record_defaults = [0.0] * (seq_length + n_forward)

        def decode_csv(row):
            # row is a string tensor containing the contents of one row
            features_list = tf.decode_csv(row, record_defaults=record_defaults)

            # getting a list of tensors
            features = features_list[:seq_length]
            labels = features_list[seq_length:]

            features = tf.stack(features)       # [SEQ_LENGTH] to shape [SEQ_LENGTH, 1]
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
            dataset = dataset.shuffle(buffer_size=10*batch_size)
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
                             batch_size=batch_size)

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
