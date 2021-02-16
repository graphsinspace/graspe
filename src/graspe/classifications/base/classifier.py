from abc import ABC, abstractmethod
from embeddings.base.embedding import Embedding


class Classifier(ABC):
    """
    Represents a single classification algorithm.
    """

    def __init__(self, embedding):
        self._data, self._labels = Embedding.from_file(embedding).get_dataset()

    @abstractmethod
    def classify(self):
        """
        Classification algorithm that must be implemented in subclasses.
        """
        pass
