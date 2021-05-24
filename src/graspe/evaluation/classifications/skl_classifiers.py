"""
scikit-learn based classifiers
"""

from sklearn import tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from evaluation.classifications.base.classifier import Classifier


class DecisionTree(Classifier):
    def __init__(self, embedding):
        super().__init__(model=tree.DecisionTreeClassifier(), embedding=embedding)


class SVM(Classifier):
    def __init__(self, embedding):
        super().__init__(model=SVC(), embedding=embedding)


class KNN(Classifier):
    def __init__(self, embedding, k_neighbors):
        super().__init__(
            model=KNeighborsClassifier(n_neighbors=k_neighbors), embedding=embedding
        )


class NaiveBayes(Classifier):
    def __init__(self, embedding):
        super().__init__(model=GaussianNB(), embedding=embedding)


class RandomForest(Classifier):
    def __init__(self, embedding, n_estimators, skip_split=False):
        super().__init__(
            model=ensemble.RandomForestClassifier(n_estimators=n_estimators),
            embedding=embedding,
            skip_split=skip_split
        )
