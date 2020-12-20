from abc import ABC, abstractmethod

from common.base.graph import Graph


class Dataset(ABC):
    """
    A model class for a graph dataset.
    """

    def __init__(self):
        self.ds = None

    @abstractmethod
    def load(self) -> Graph:
        pass
