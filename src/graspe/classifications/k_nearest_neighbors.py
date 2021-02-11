from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np

from classifications.base.classifier import Classifier

class KNN(Classifier):
    def __init__(self, embedding, k_neighbors):
        super().__init__(embedding)
        self._k_neighbors = k_neighbors

    def classify(self):

        train_data, test_data, train_labels, test_labels = train_test_split(
            self._data, self._labels, test_size=0.2
        )

        knn = KNeighborsClassifier(n_neighbors=self._k_neighbors)

        knn.fit(train_data, train_labels)

        predicted_labels = knn.predict(test_data)

        acc = accuracy_score(test_labels, predicted_labels)
        print("Accuracy", acc)
        print(
            "Precisions: ", precision_score(test_labels, predicted_labels, average=None)
        )
        print("Recalls: ", recall_score(test_labels, predicted_labels, average=None))
