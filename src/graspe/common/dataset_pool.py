import os
import networkx as nx
from common.graph_loaders import load_from_file
from common.graph import Graph


class DatasetPool:
    """
    Class that enables loading of various graph-based datasets.
    """

    __pool = None

    @staticmethod
    def load(name):
        """
        Loads the graph-based dataset of the given name.

        Parameters
        ----------
        name : string
            Name of the dataset.

        Returns the loaded graph.
        """
        DatasetPool.__init_pool()
        if name in DatasetPool.__pool:
            method, parameter = DatasetPool.__pool[name]
            return method(parameter)
        return None

    @staticmethod
    def get_datasets():
        """
        Returns names of the available datasets.
        """
        DatasetPool.__init_pool()
        return DatasetPool.__pool.keys()

    @staticmethod
    def __init_pool():
        """
        Initializes dataset pool.
        """
        if DatasetPool.__pool != None:
            return
        DatasetPool.__pool = {}

        # Init from "data" directory.
        file_dataset_labels = {
            "amazon_electronics_computers": "labels",
            "amazon_electronics_photo": "labels",
            "citeseer": "labels",
            "cora_ml": "labels",
            "cora": "labels",
            "dblp": "labels",
            "pubmed": "labels",
        }
        file_dataset_needs_dense = [
            "cora_ml",
            "cora",
            "amazon_electronics_computers",
            "dblp",
            "amazon_electronics_photo",
            "citeseer",
            "pubmed",
        ]
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "data"
        )
        for f in os.listdir(base_path):
            path = os.path.join(base_path, f)
            if os.path.isfile(path) and f[0] != ".":
                name, ext = os.path.splitext(f)
                DatasetPool.__pool[name] = (
                    lambda x: load_from_file(
                        x,
                        file_dataset_labels.get(
                            os.path.splitext(os.path.basename(x))[0]
                        ),
                        to_dense=os.path.splitext(os.path.basename(x))[0]
                        in file_dataset_needs_dense,
                    ),
                    path,
                )

        # Init form networkx
        nx_dataset_labels = {
            "karate_club_graph": "club",
            "davis_southern_women_graph": None,
            "florentine_families_graph": None,
            "les_miserables_graph": None,
        }
        for dataset in nx_dataset_labels:
            DatasetPool.__pool[dataset] = (
                lambda x: Graph(getattr(nx, x)(), nx_dataset_labels[x]),
                dataset,
            )
