from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from embeddings.base.embedding import Embedding


class Classifier:
    """
    Represents a single classification algorithm with methods for fit, predict and metrics.
    """

    def __init__(self, model, embedding):
        self._data, self._labels = Embedding.from_file(embedding).get_dataset()
        self.model = model
        self.predicted_labels = None
        self.fitted = False
        (
            self.train_data,
            self.test_data,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(self._data, self._labels, test_size=0.33)

    def fit(self):
        self.model.fit(self.train_data, self.train_labels)
        self.fitted = True

    def predict(self):
        self.predicted_labels = self.model.predict(self.test_data)
        return self.predicted_labels

    def fit_predict(self):
        self.fit()
        return self.predict()

    def accuracy(self):
        acc = accuracy_score(self.test_labels, self.predicted_labels)
        print("Accuracy: ", acc)
        return acc

    def precision(self):
        prec = precision_score(self.test_labels, self.predicted_labels, average=None)
        print("Precision: ", prec)
        return prec

    def recall(self):
        recall = recall_score(self.test_labels, self.predicted_labels, average=None)
        print("Recall: ", recall)
        return recall

    # Override this for custom non-sklearn methods:
    def classify(self):
        self.fit_predict()

        return self.accuracy(), self.precision(), self.recall()
