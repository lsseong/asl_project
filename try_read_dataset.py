from model import model
import tensorflow as tf


if __name__ == '__main__':
    seq_length = 8
    n_forward = 5

    with tf.Session() as sess:
        fn = model.read_dataset(filename="data/train_{}_{}.csv".format(seq_length, n_forward),
                                mode=tf.estimator.ModeKeys.TRAIN,
                                seq_length=seq_length,
                                n_forward=n_forward,
                                batch_size=100)

        batch_features, batch_labels = fn()
        features, labels = sess.run([batch_features, batch_labels])
        print("try_out_input_function: features shape = {}".format(features['prices'].shape))
        print("try_out_input_function: labels shape = {}".format(labels.shape))
