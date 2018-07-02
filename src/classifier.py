import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d


class Classifier:
    def __init__(self, decider=None):
        self.decider = decider
        if self.decider is None:
            self.decider = SVC()

    def fit(self, x, y):
        self._fit_low_quality_classifier(x, pd.DataFrame(y, copy=True))
        self._fit_low_quality_predictor(x, pd.DataFrame(y, copy=True))
        self._fit_mid_quality_classifier(x, pd.DataFrame(y, copy=True))
        self._fit_mid_quality_predictor(x, pd.DataFrame(y, copy=True))
        self._fit_high_quality_classifier(x, pd.DataFrame(y, copy=True))
        self._fit_high_quality_predictor(x, pd.DataFrame(y, copy=True))

        dataset = self._preprocess(x)

        dataset2 = pd.concat([dataset, y], axis=1)
        dataset2.to_csv('../data/newdata.csv', index=False)

        self.decider.fit(dataset, y)

    def predict(self, x):
        dataset = self._preprocess(x)

        # pd.concat([dataset, y], axis=1).to_csv('../data/new_data.csv', index=False)

        return self.decider.predict(dataset)

    def _preprocess(self, x):
        is_lq = self.lqc.predict(x)
        is_mq = self.mqc.predict(x)
        is_hq = self.hqc.predict(x)

        lq_guess = self.lqp.predict(x)
        mq_guess = self.mqp.predict(x)
        hq_guess = self.hqp.predict(x)

        return pd.concat([x, pd.DataFrame({'is_low_q': is_lq, 'is_mid_q': is_mq, 'is_high_q': is_hq,
                                           'low_q_guess': lq_guess, 'mid_q_guess': mq_guess,
                                           'high_q_guess': hq_guess}, index=x.index)]
                         , axis=1)

    def _fit_low_quality_classifier(self, x, y):
        # 0.9590620641562065    RandomForestClassifier(n_estimators=15, max_depth=5, max_features=0.9, random_state=1)
        _y = (y.quality < 5).values.ravel()

        self.lqc = RandomForestClassifier(n_estimators=15, max_depth=5, max_features=0.9, random_state=1)
        self.lqc.fit(x, _y)

    def _fit_low_quality_predictor(self, x, y):
        # 0.832777777777778     SVC(c>=0.5, g>=5, class_weight='balanced', random_state=1)

        indices = y.quality < 5
        _x, _y = x[indices], y.quality[indices]

        self.lqp = SVC(C=20, kernel='rbf', gamma=10, class_weight='balanced', random_state=1)
        self.lqp.fit(_x, _y)

    def _fit_mid_quality_classifier(self, x, y):
        # 0.8680152186397571        KNeighborsClassifier(n_neighbors=66, weights='distance')
        _y = (y.quality == 5) | (y.quality == 6)

        self.mqc = KNeighborsClassifier(n_neighbors=66, weights='distance')
        # print(cross_val_score(self.mqc, x, _y, scoring='f1_micro', cv=5).mean())
        self.mqc.fit(x, _y)

    def _fit_mid_quality_predictor(self, x, y):
        # RES: 0.7722504230118444       KNeighborsClassifier(n_neighbors=65, weights='distance')
        indices = (y.quality == 5) | (y.quality == 6)
        _x, _y = x[indices], y.quality[indices]

        self.mqp = KNeighborsClassifier(n_neighbors=65, weights='distance')
        # print(cross_val_score(self.mqp, _x, _y, scoring='f1_micro', cv=5).mean())
        self.mqp.fit(_x, _y)

    def _fit_high_quality_classifier(self, x, y):
        # 0.9039156206415621        KNeighborsClassifier(n_neighbors=70, weights='distance')
        # RandomForestClassifier(n_estimators=15, max_depth=20, max_features=0.8, random_state=1)
        _y = (y.quality > 6)

        self.hqc = RandomForestClassifier(n_estimators=15, max_depth=20, max_features=0.8, random_state=1)
        self.hqc.fit(x, _y)

    def _fit_high_quality_predictor(self, x, y):
        # RES: 0.932172531769306        KNearestNeigbors(n_neighbors>8, weights='distance')
        # RES: 0.932172531769306        SVC(C=1, gamma=20, class_weight='balanced', random_state=1)
        indices = y.quality > 6
        _x, _y = x[indices], y.quality[indices]
        self.hqp = SVC(C=1, gamma=20, class_weight='balanced', random_state=1)
        # print(cross_val_score(self.hqp, _x, _y, scoring='f1_micro', cv=5).mean())
        self.hqp.fit(_x, _y)
