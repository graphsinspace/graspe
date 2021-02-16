from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np

from classifications.base.classifier import Classifier


class RandomForest(Classifier):
    def __init__(self, embedding, n_estimators):
        super().__init__(embedding)
        self._n_estimators = n_estimators

    def classify(self):

        train_data, test_data, train_labels, test_labels = train_test_split(
            self._data, self._labels, test_size=0.2
        )

        rf = ensemble.RandomForestClassifier(n_estimators=self._n_estimators)
        rf.fit(train_data, train_labels)

        predicted_labels = rf.predict(test_data)

        acc = accuracy_score(test_labels, predicted_labels)
        print("Accuracy", acc)
        print(
            "Precisions: ", precision_score(test_labels, predicted_labels, average=None)
        )
        print("Recalls: ", recall_score(test_labels, predicted_labels, average=None))
