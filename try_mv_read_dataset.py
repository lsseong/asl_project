from model import mvmodel
import tensorflow as tf


if __name__ == '__main__':
    n_factor = 2
    seq_length = 5
    n_forward = 2
    batch_size = 100

    with tf.Session() as sess:

        filename = "data/train_multi_close_{}_{}_{}.csv".format(n_factor, seq_length, n_forward)

        fn = mvmodel.read_dataset(filename=filename,
                                  mode=tf.estimator.ModeKeys.TRAIN,
                                  n_factor=n_factor,
                                  seq_length=seq_length,
                                  n_forward=n_forward,
                                  batch_size=batch_size)

        batch_features, batch_labels = fn()
        features, labels = sess.run([batch_features, batch_labels])
        print("try_out_input_function: features shape = {}".format(features['prices'].shape))
        print("try_out_input_function: labels shape = {}".format(labels.shape))
