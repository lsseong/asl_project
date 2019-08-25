import tensorflow as tf
import numpy as np
import pandas as pd

n_factor = 2
seq_length = 5
num_features = n_factor * seq_length

SAVED_MODEL = "c:/tf/uob/seq2ret/export/exporter/" + "1566303982"
PREDICT_FILE = "data/train_multi_seq2ret_{}_{}_1.csv".format(n_factor, seq_length)

# load prediction function
predict_fn = tf.contrib.predictor.from_saved_model(SAVED_MODEL)

# load data for prediction
df = pd.read_csv(PREDICT_FILE, delimiter=",", header=None)
features = df.iloc[:, 0:num_features].to_numpy()
features = np.reshape(features, (-1, seq_length, n_factor))
actuals = df.iloc[:, num_features:].to_numpy()

# get predictions
predicts = predict_fn({"prices": features})
predicts = predicts['predicted']
predicts = np.reshape(predicts, (-1, 1))

print(actuals.shape)
print(predicts.shape)

# create output data frame
output_headers = [('a{}'.format(i)) for i in range(1)]
output_headers = output_headers + [('p{}'.format(i)) for i in range(1)]
output = pd.DataFrame(np.concatenate((actuals, predicts), axis=1), columns=output_headers)

print(output.head())

# compute directional accuracy
accuracy = np.sum(np.sign(output['a0'] * output['p0']) > 0) / len(output)
print(accuracy)

# save to file
output_filename = "c:/tf/predict/uob/predict_seq2ret_{}_{}.csv".format(n_factor, seq_length)
output.to_csv(output_filename, float_format="%.4f")
