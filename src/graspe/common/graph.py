class Graph:
    """
    A model class for graph.

    Attributes
    ----------
    __graph : dict
        Data structure that stores all the graph's nodes and edges.
    """

    def __init__(self):
        """
        Parameters
        ----------
        """
        self.__graph = {}

    def add_node(self, id, label=''):
        """
        Adds a new node into the graph.

        Parameters
        ----------
        id : ?
            The node's identifier.
        label : ?
            The node's label (class).
        """
    
    def add_edge(self, node1, node2, weight=0):
        """
        Adds a new edge into the graph.

        Parameters
        ----------
        node1 : ?
            Identifier of the edge's starting node.
        node2 : ?
            Identifier of the edge's ending node.
        weight: numeric
            Weight of the edge.
        """
    
    def induce_by_random_nodes(self, p):
        """
        Generates a graph induced by p*|N| randomly selected nodes.

        Parameters
        ----------
        p : float
            A value in the range (0,1]. Determines the size of the resulting graph's nodes set.
            The resulting graph's nodes set will have p*|N| randomly selected nodes.
        """
        return Graph()
    
    def induce_by_random_edges(self, p):
        """
        Generates a graph induced by p*|E| randomly selected edges.

        Parameters
        ----------
        p : float
            A value in the range (0,1]. Determines the size of the resulting graph's edges set.
            The resulting graph's edges set will have p*|E| randomly selected edges.
        """
        return Graph()

    def link_precision(self, g):
        """
        The fraction of correct links in g.
        A link in g is considered correct if it is also present within the links of the current graph object (self).

        Parameters
        ----------
        g : common.graph.Graph
            A graph object.
        """
        return 0

    def map(self, g):
        """
        MAP estimates precision for every node and computes the average over all nodes.

        Parameters
        ----------
        g : common.graph.Graph
            A graph object.
        """
        return 0