"""
We can use linear regression to compare the weights with those obtain from Tensorflow
"""

import pandas as pd
import statsmodels.formula.api as sm

filename = 'data/train_3_1.csv'

df = pd.read_csv(filename, delimiter=",", header=None)
df.columns = ['t1', 't2', 't3', 't4']

print(df.head())

result = sm.ols(formula="t4 ~ t1 + t2 + t3", data=df).fit()

print(result.params)

print(result.summary())