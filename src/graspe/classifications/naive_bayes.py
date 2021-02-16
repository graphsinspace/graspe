from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np

from classifications.base.classifier import Classifier


class NaiveBayes(Classifier):
    def classify(self):

        train_data, test_data, train_labels, test_labels = train_test_split(
            self._data, self._labels, test_size=0.33
        )

        gnb = GaussianNB()
        gnb.fit(train_data, train_labels)

        predicted_labels = gnb.predict(test_data)

        acc = accuracy_score(test_labels, predicted_labels)
        print("Accuracy", acc)
        print(
            "Precisions: ", precision_score(test_labels, predicted_labels, average=None)
        )
        print("Recalls: ", recall_score(test_labels, predicted_labels, average=None))
