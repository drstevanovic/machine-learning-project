import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

def read_file(file):
    """
    Reads a csv file and returns pandas dataframes
    :param file: csv file
    :return: x, y: input columns and output column
    """
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                       'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    data = pd.read_csv(file)
    print(data.head())
    x = data[ [*feature_columns, 'quality'] ]
    y = data[["quality"]]
    print(y)
    return x


def create_train_test_files(dataset_file, train_path, test_path):
    """
    Splits data from dataset_file into train and test maintaining distribution of classes
    and writes them to separate files.
    :param dataset_file: input file
    :param train_path: train data will be written to this file
    :param test_path: test data will be written in this file
    """
    x, y = read_file(dataset_file)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, shuffle=True,
                                                        stratify=y)

    train_set = pd.concat([x_train, y_train], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)


def test_model(train_file):
    data = read_file(train_file)

    # First center and scale the data
    scaled_data = preprocessing.scale(data) # scale funkcija ocekuje da torke budu u redovima, ne kolonama

    pca = PCA()  # create a PCA object
    pca.fit(scaled_data)  # do the math
    pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data
    print(pca_data)

    # The following code constructs the Scree plot
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    # Determine which vine properties had the biggest influence on PC1
    ## get the name of the top 10 measurements (vine properties) that contribute
    ## most to pc1.
    ## first, get the loading scores
    loading_scores = pd.Series(pca.components_[0], index=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                       'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])
    ## now sort the loading scores based on their magnitude
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

    # get the names of the top 10 vine properties
    top_10_genes = sorted_loading_scores[0:10].index.values

    ## print the gene names and their scores (and +/- sign)
    print(loading_scores[top_10_genes])
    # sa > https://statquest.org/2018/01/08/statquest-pca-in-python/


def main():
    train_path = '../data/train.csv'
    test_path = '../data/test.csv'
    # main(train_path, test_path)
    test_model(train_path)
    # create_train_test_files("../data/winequality-red.csv", train_path, test_path)


if __name__ == '__main__':
    main()
