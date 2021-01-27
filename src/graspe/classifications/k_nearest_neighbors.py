from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np


class KNN():
    def __init__(
        self,
        g,
        embedding,
        k_neighbors
    ):
        self._g = g
        self._embedding = embedding
        self._k_neighbors = k_neighbors


    def classify(self):

        nodes = self._g.nodes()

        labels = [n[1]['label'] for n in nodes]

        embedding_file = open(self._embedding, 'r')
        lines = embedding_file.readlines()

        node_vectors = []
        for line in lines:
            line = line.split(":")[1]
            line = line.split(",")
            
            line = np.array(line, dtype='float64')
            node_vectors.append(line)


        train_data, test_data, train_labels, test_labels = train_test_split(node_vectors, labels, test_size=.2)

        knn = KNeighborsClassifier(n_neighbors=self._k_neighbors)

        knn.fit(train_data, train_labels)

        predicted_labels = knn.predict(test_data)

        acc = accuracy_score(test_labels, predicted_labels)
        print("Accuracy", acc)
        print("Precisions: ", precision_score(test_labels, predicted_labels, average=None))
        print("Recalls: ", recall_score(test_labels, predicted_labels, average=None))
