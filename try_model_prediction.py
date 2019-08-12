"""
Load a trained model and do prediction, check accuracy

"""
import tensorflow as tf
import numpy as np
import pandas as pd

n_factor = 2
seq_length = 5
n_forward = 2
model = "linear"
num_features = n_factor * seq_length

SAVED_MODEL = "c:/tf/uob/{}/export/exporter/".format(model) + "1565616086"
PREDICT_FILE = "data/train_multi_close_{}_{}_{}.csv".format(n_factor, seq_length, n_forward)

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

print(actuals.shape)
print(predicts.shape)

# create output data frame
output_headers = [('a{}'.format(i)) for i in range(n_forward)]
output_headers = output_headers + [('p{}'.format(i)) for i in range(n_forward)]
output = pd.DataFrame(np.concatenate((actuals, predicts), axis=1), columns=output_headers)


# compute directional accuracy by comparing price change between start and ending prices
def price_change(row, start_label, end_label):
    if row[end_label] > row[start_label]:
        return 1
    elif row[end_label] < row[start_label]:
        return -1
    else:
        return 0


output['a_change'] = output.apply(lambda row: price_change(row, 'a0', 'a{}'.format(n_forward-1)), axis=1)
output['p_change'] = output.apply(lambda row: price_change(row, 'p0', 'p{}'.format(n_forward-1)), axis=1)
output['correct_change'] = (output['a_change'] == output['p_change']).astype(int)

print(output.head())

accuracy = output['correct_change'].values.sum() / len(output)
print(accuracy)

# save to file
output_filename = "c:/tf/predict/uob/predict_{}_{}_{}.csv".format(model, n_factor, seq_length)
output.to_csv(output_filename, float_format="%.4f")




