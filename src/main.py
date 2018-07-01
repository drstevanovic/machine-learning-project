import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


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
    x = data[feature_columns]
    y = data[["quality"]]
    print(y)
    return x, y


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
    x, y = read_file(train_file)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=1)
    for train_index, test_index in sss.split(x, y):
        X_train, X_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        # print(y_train.value_counts(sort=False, normalize=True))
        print(test_index)
        # print(y_test.value_counts(sort=False, normalize=True))
        print()


def main():
    train_path = '../data/train.csv'
    test_path = '../data/test.csv'
    # main(train_path, test_path)
    # test_model(train_path)
    create_train_test_files("../data/winequality-red.csv", train_path, test_path)


if __name__ == '__main__':
    main()
