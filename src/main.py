import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def test_model(train_file):
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                       'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    train_set = pd.read_csv(train_file)

    print(train_set.corr())
    print("Ovo crtanje traje dugo")
    sbn.pairplot(train_set, hue="quality")
    plt.show()

    x = train_set[feature_columns]
    y = train_set["quality"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

    cls = LogisticRegression(C=100)
    cls.fit(x_train, y_train)
    y_pred = cls.predict(x_test)
    print(accuracy_score(y_test, y_pred))


def main(train_file, test_file):
    pass


if __name__ == '__main__':
    train_path = '../data/winequality-red.csv'
    test_path = '../data/test.csv'
    # main(train_path, test_path)
    test_model(train_path)
