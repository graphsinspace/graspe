from common.graph import Graph

def load_from_file(path):
    name, ext = os.path.splitext(path)
    f = globals().get("load_"+ext[1:])
    if f == None:
        raise Exception('loader for the extension {} is not implemented'.format(ext[1:]))
    return f(path)

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

def load_npz(path):
    """
    Loads a graph from a npz file.

    Parameters
    ----------
    path : string
        Path to the npz file.

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    return Graph()