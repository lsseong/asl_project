"""
This module implements multivariate (multiple factors) sequence to sequence prediction models.

"""
import tensorflow as tf

TIME_SERIES_INPUT = "prices"


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

        # features = [BATCH_SIZE, SEQ_LENGTH]
        # labels = [BATCH_SIZE, N_FORWARD]
        return features, labels

    return _input_fn


