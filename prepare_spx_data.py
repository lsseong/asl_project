from preprocess import fileutils
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_label(row):
    value = row['max_drawdown_percent']

    if value <= 0.05:
        return 0
    elif value <= 0.10:
        return 1
    else:
        return 2


if __name__ == '__main__':
    df_inputs = fileutils.read_csv('spx/spx.csv', '%m/%d/%Y')
    df_labels = fileutils.read_csv('spx/spx_labels.csv', '%m/%d/%Y')

    # create labels
    y = df_labels[['max_drawdown_percent']]
    y['label'] = y.apply(lambda row: create_label(row), axis=1)

    # prepare features
    feature_cols = ['Z FX Vol', 'Z VIX', 'Z IG Spreads']
    X = df_inputs[feature_cols]
    X = X[X.index.isin(y.index)]

    print(X.head())
    print(y.head())

    X.to_csv('spx/spx_X.csv')
    y.to_csv('spx/spx_y.csv')

    del y['max_drawdown_percent']

    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # create decision tree classifier
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)

    # train classifier
    clf = clf.fit(X_train, y_train)

    # predict the response for the test dataset
    y_pred = clf.predict(X_test)

    # model accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # graph decision tree
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('spx/tree.png')
    Image(graph.create_png())

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm, index=['0', '1', '2'], columns=['0', '1', '2'])

    plt.figure(figsize=(5.5, 4))
    sns.heatmap(cm_df, annot=True, fmt='d')
    plt.title('Decision tree \nAccuracy:{0:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
