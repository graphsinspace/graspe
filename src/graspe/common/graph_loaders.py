import os
from common.graph import Graph

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
    f = globals().get("load_"+ext[1:])
    if f == None:
        raise Exception('loader for the extension {} is not implemented'.format(ext[1:]))
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
    raise Exception('Not implemented')
    return Graph()

def load_npz(path, label):
    """
    Loads a graph from a npz file.

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
    raise Exception('Not implemented')
    return Graph()