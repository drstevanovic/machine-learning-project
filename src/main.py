import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import *


def read_file(file):
    """
    Reads a csv file and returns pandas dataframes
    :param file: csv file
    :return: x, y: input columns and output column
    """
    feature_columns = [
        'alcohol',  # 0.49
        'volatile acidity',  # -0.38
        'sulphates',  # 0.27
        'citric acid',  # 0.23
        'total sulfur dioxide',  # -0.19
        'density',  # -0.19
        'chlorides',  # -0.13
        'fixed acidity',  # 0.11
        'free sulfur dioxide',  # -0.058
        'pH',  # -0.052
        'residual sugar'  # -0.0004
    ]

    feature_columns = [
        'alcohol',  # 0.49
        'volatile acidity',  # -0.38
        'sulphates',  # 0.27
        'citric acid',  # 0.23
        'total sulfur dioxide',  # -0.19
        'density',  # -0.19
        'chlorides',  # -0.13
        'fixed acidity'  # 0.11
    ]

    data = pd.read_csv(file)
    x = data[feature_columns]
    # x = data[ [*feature_columns, 'quality'] ]
    y = data[["quality"]]
    return x, y


# Random forest
def low_quality_classifier(x, y):
    # RES:0.9590620641562065    RandomForestClassifier(n_estimators=15, max_depth=5, max_features=0.9, random_state=1)

    print("LOW QUALITY CLASSIFIER")

    y = y.quality < 5

    cls = RandomForestClassifier(n_estimators=15, max_depth=5, max_features=0.9, random_state=1)
    print(cross_val_score(cls, x, y, scoring='f1_micro', cv=5).mean())


# SVC
def low_quality_predictor(x, y):
    # 0.832777777777778     SVC(c>=0.5, g>=5, class_weight='balanced', random_state=1)
    print("LOW QUALITY PREDICTOR")
    indices = y.quality < 5
    x, y = x[indices], y.quality[indices]

    cls = SVC(C=20, kernel='rbf', gamma=10, class_weight='balanced', random_state=1)
    print(cross_val_score(cls, x, y, scoring='f1_micro', cv=5).mean())


# KNN
def mid_quality_classifier(x, y):
    # 0.8680152186397571        KNeighborsClassifier(n_neighbors=66, weights='distance')
    print("MID QUALITY CLASSIFIER")
    y = (y.quality == 5) | (y.quality == 6)

    cls = KNeighborsClassifier(n_neighbors=66, weights='distance')
    print(cross_val_score(cls, x, y, scoring='f1_micro', cv=5).mean())


# KNN
def mid_quality_predictor(x, y):
    # RES: 0.7722504230118444       KNeighborsClassifier(n_neighbors=65, weights='distance')
    print("MID QUALITY PREDICTOR")
    indices = (y.quality == 5) | (y.quality == 6)
    x, y = x[indices], y.quality[indices]

    cls = KNeighborsClassifier(n_neighbors=65, weights='distance')
    print(cross_val_score(cls, x, y, scoring='f1_micro', cv=5).mean())


# Random forest
def high_quality_classifier(x, y):
    # 0.9039156206415621        KNeighborsClassifier(n_neighbors=70, weights='distance')
    # RandomForestClassifier(n_estimators=15, max_depth=20, max_features=0.8, random_state=1)
    print("HIGH QUALITY CLASSIFIER")
    y = (y.quality > 6)

    cls = RandomForestClassifier(n_estimators=15, max_depth=20, max_features=0.8, random_state=1)
    print(cross_val_score(cls, x, y, scoring='f1_micro', cv=5).mean())


# SVC
def high_quality_predictor(x, y):
    # RES: 0.932172531769306        KNearestNeigbors(n_neighbors>8, weights='distance')
    # RES: 0.932172531769306        SVC(C=1, gamma=20, class_weight='balanced', random_state=1)
    print("HIGH QUALITY PREDICTOR")
    indices = y.quality > 6
    x, y = x[indices], y.quality[indices]

    cls = SVC(C=1, gamma=20, class_weight='balanced', random_state=1)
    print(cross_val_score(cls, x, y, scoring='f1_micro', cv=5).mean())


def try_svc(x, y):
    print("SVC")
    cs = [0.1, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000, 50000]
    gammas = [0.000001, 0.01, 0.1, 0.2, 0.5, 0.7, 1, 1.3, 2, 5, 8, 10, 20, 100, 1000, 1000000]
    scores = []
    for c in cs:
        for gamma in gammas:
            svc = SVC(C=c, gamma=gamma, class_weight='balanced', random_state=1)
            score = cross_val_score(svc, x, y, scoring='f1_micro', cv=5).mean()
            print("C:", c, ", gamma:", gamma, ", score:", score)
            scores.append(score)
    print(max(scores))


def try_knn(x, y):
    print("K Nearest Neighbors")
    ns = [3, 5, 8, 10,
          12, 15, 20, 24, 28,
          35, 43, 50, 58, 65, 70,
          82, 95, 100, 120]

    scores = []
    for n in ns:
        knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
        score = cross_val_score(knn, x, y, scoring='f1_micro', cv=5).mean()
        print("N:", n, ", score:", score)
        scores.append(score)
    print(max(scores))


def try_rf(x, y):
    print("RANDOM FOREST")
    ns = [5, 8, 10, 12, 15, 18, 23, 30, 35, 50, 58, 65, 70, 82, 95, 100, 120, 150, 200]
    max_d = [5, 10, 20, 50, 100]
    max_feat = [0.7, 0.8, 0.9]

    scores = []
    for n in ns:
        for d in max_d:
            for feat in max_feat:
                cls = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=feat, random_state=1)
                score = cross_val_score(cls, x, y, scoring='f1_micro', cv=5).mean()
                print("N: {}, d: {}, feat: {},   Score: {}".format(n, d, feat, score))
                scores.append(score)
    print(max(scores))


def test_model(train_path):
    x, y = read_file(train_path)

    x_scaler = StandardScaler()
    x = x_scaler.fit_transform(x)

    low_quality_classifier(x, y)
    low_quality_predictor(x, y)
    mid_quality_classifier(x, y)
    mid_quality_predictor(x, y)
    high_quality_classifier(x, y)
    high_quality_predictor(x, y)


def do_pca(train_file):
    x, y = read_file(train_file)

    # First center and scale the data
    scaled_data = preprocessing.scale(x)  # scale funkcija ocekuje da torke budu u redovima, ne kolonama

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
    loading_scores = pd.Series(pca.components_[0],
                               index=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                                      'free sulfur dioxide',
                                      'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
    ## now sort the loading scores based on their magnitude
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

    # get the names of the top 10 vine properties
    top_10_properties = sorted_loading_scores[0:10].index.values

    ## print the gene names and their scores (and +/- sign)
    print(loading_scores[top_10_properties])
    # sa > https://statquest.org/2018/01/08/statquest-pca-in-python/


def main():
    train_path = '../data/train.csv'
    test_path = '../data/test.csv'

    test_model(train_path)


if __name__ == '__main__':
    main()
