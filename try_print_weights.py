"""
Load saved model and print the weights

"""

import tensorflow as tf
import tensorflow.saved_model.tag_constants as tag_constants

export_dir = 'C:/tf/energy/linear/export/exporter/1564888458'

with tf.Session() as sess:
  tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
  vars = tf.trainable_variables()
  print(vars)

  vars_vals = sess.run(vars)

  for var, val in zip(vars, vars_vals):
    print("var: {}, value: {}".format(var.name, val))
