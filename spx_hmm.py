from preprocess import fileutils

import datetime
import pickle
import warnings

# conda install -c conda-forge hmmlearn
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import numpy as np
import pandas as pd
import seaborn as sns


def plot_in_sample_hidden_states(hmm_model_, df_, rets_):
    """
    Plot the SPX prices masked by
    the in-sample hidden states as a mechanism
    to understand the market regimes.
    """
    # Predict the hidden states array
    hidden_states = hmm_model_.predict(rets_)
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        hmm_model.n_components,
        sharex=True, sharey=True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, hmm_model.n_components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax.plot_date(
            df_.index[mask],
            df_["SPX"][mask],
            ".", linestyle='none',
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()


if __name__ == '__main__':
    # Hides deprecation warnings for sklearn
    warnings.filterwarnings("ignore")

    df = fileutils.read_csv('spx/spx.csv', '%m/%d/%Y')
    df["Returns"] = df["SPX"].pct_change()
    df.dropna(inplace=True)

    rets = np.column_stack([df["Returns"]])

    # Create the Gaussian Hidden markov Model and fit it
    # to the SPY returns data, outputting a score
    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(rets)
    print("Model Score:", hmm_model.score(rets))

    # Plot the in sample hidden states closing values
    plot_in_sample_hidden_states(hmm_model, df, rets)
