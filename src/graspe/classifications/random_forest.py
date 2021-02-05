from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np


class RandomForest():
    def __init__(
        self,
        g,
        embedding,
        n_estimators
    ):
        self._g = g
        self._embedding = embedding
        self._n_estimators = n_estimators

    
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

        rf = ensemble.RandomForestClassifier(n_estimators=self._n_estimators)
        rf.fit(train_data, train_labels)

        predicted_labels = rf.predict(test_data)

        acc = accuracy_score(test_labels, predicted_labels)
        print("Accuracy", acc)
        print("Precisions: ", precision_score(test_labels, predicted_labels, average=None))
        print("Recalls: ", recall_score(test_labels, predicted_labels, average=None))
