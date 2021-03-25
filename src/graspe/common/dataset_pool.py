import os

import networkx as nx
import numpy as np
import scipy.sparse

from common.graph import Graph
from common.graph_loaders import load_from_file


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
                name, _ = os.path.splitext(f)
                DatasetPool.__pool[name] = (
                    lambda x: load_from_file(
                        x,
                        file_dataset_labels.get(
                            os.path.splitext(os.path.basename(x))[0], "labels"
                        ),
                        to_dense=os.path.splitext(os.path.basename(x))[0]
                        in file_dataset_needs_dense,
                    ),
                    path,
                )

        # Init from networkx
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

    @staticmethod
    def generate_random_graphs(n_vals, k_vals, out):
        graphs = {}
        for n in n_vals:
            p_vals = [k / (n - 1) for k in k_vals]
            for p in p_vals:
                graphs["erdos-renyi_n{}_p{}".format(n, p)] = nx.fast_gnp_random_graph(
                    n, p
                )
                for k in k_vals:
                    graphs[
                        "newman-watts-strogatz_n{}_p{}_k{}".format(n, p, k)
                    ] = nx.newman_watts_strogatz_graph(n, k, p)
                    graphs[
                        "powerlaw-cluster_n{}_m{}_p{}".format(n, k, p)
                    ] = nx.powerlaw_cluster_graph(n, k, p)
                    name = "barabasi-albert_n{}_m{}".format(n, k)
                    if not name in graphs:
                        graphs[name] = nx.barabasi_albert_graph(n, k)
        for g in graphs:
            csr = scipy.sparse.csr_matrix(nx.to_scipy_sparse_matrix(graphs[g]))
            np.savez(
                os.path.join(out, g + ".npz"),
                adj_data=csr.data,
                adj_indices=csr.indices,
                adj_indptr=csr.indptr,
                adj_shape=csr.shape,
                labels=[],
            )
