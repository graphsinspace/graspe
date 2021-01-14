import os
from common.graph import Graph
import numpy as np
import scipy
import scipy.sparse as sp
import networkx as nx


def load_from_file(path, label):
    """
    Loads a graph from a file.

    Parameters
    ----------
    path : string
        Path to the file.
    label: string
        Name of node label.

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    name, ext = os.path.splitext(path)
    f = globals().get("load_" + ext[1:])
    if f == None:
        raise Exception(
            "loader for the extension {} is not implemented".format(ext[1:])
        )
    return f(path, label)


def load_csv(path, label):
    """
    Loads a graph from a CSV file.

    Parameters
    ----------
    path : string
        Path to the CSV file.
    label: string
        Name of node label.

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    raise Exception("Not implemented")
    return Graph()


def load_npz(path, label="labels"):
    """
    Loads a graph from a npz file. For included npz files see:
    https://github.com/abojchevski/graph2gauss/blob/master/g2g/utils.py#L479

    Parameters
    ----------
    path : string
        Path to the npz file.
    label: string
        Name of node label.

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    with np.load(path) as loader:
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        attr_matrix = sp.csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=loader["attr_shape"],
        )

        labels = loader.get(label)

        nx_graph = nx.from_scipy_sparse_matrix(adj_matrix)

        # get labels first
        node_attrs = {i: {"label": label} for i, label in enumerate(labels)}

        # add attrs
        for node_id in node_attrs:
            node_attrs[node_id]["attrs"] = attr_matrix[node_id]

        nx.set_node_attributes(nx_graph, node_attrs)

        return Graph(nx_graph)
