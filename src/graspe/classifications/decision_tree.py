from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np

from classifications.base.classifier import Classifier

class DecisionTree(Classifier):

    def classify(self):

        train_data, test_data, train_labels, test_labels = train_test_split(
            self._data, self._labels, test_size=0.33
        )

        clf = tree.DecisionTreeClassifier()
        clf.fit(train_data, train_labels)

        predicted_labels = clf.predict(test_data)

        acc = accuracy_score(test_labels, predicted_labels)
        print("Accuracy", acc)
        print(
            "Precisions: ", precision_score(test_labels, predicted_labels, average=None)
        )
        print("Recalls: ", recall_score(test_labels, predicted_labels, average=None))
