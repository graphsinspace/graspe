# TODO: remove this, only use Dataset as one dataset can have many graphs?

from common.base.graph import Graph


def load_csv(path):
    """
    Loads a graph from a CSV file.

    Parameters
    ----------
    path : string
        Path to the CSV file.

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    return Graph()
