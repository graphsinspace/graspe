import dgl
import networkx as nx
import random

class Graph:
    """
    A model class for graph.

    Attributes
    ----------
    __graph : dict
        Data structure that stores all the graph's nodes and edges.
    """

    def __init__(self, data=None, label=None):
        """
        Parameters
        data: ?
            Data to initialize graph. If None (default) an empty graph is created. 
            Can be an edge list, NumPy matrix, 2D array, SciPy sparse matrix, GRASPE graph, NetworkX graph, PyGraphviz graph, DGL graph. 
        ----------
        """
        if (isinstance(data, Graph)):
            self.__graph = nx.DiGraph(data.__graph)
        else:
            if (isinstance(data, dgl.DGLGraph)):
                self.__graph = data.to_networkx()
            else:
                self.__graph = nx.DiGraph(data)
            if label:
                mapping = {}
                old_labels = nx.get_node_attributes(self.__graph, label)
                new_labels = {}
                for node_id in old_labels:
                    old_label = old_labels[node_id] 
                    if not old_label in mapping:
                        mapping[old_label] = len(mapping)
                    new_labels[node_id] = mapping[old_label]
                nx.set_node_attributes(self.__graph, new_labels, 'label')

    def nodes(self):
        """
        Returns all nodes of the graph.
        """
        return list(self.__graph.nodes(data=True))

    def edges(self):
        """
        Returns all edges of the graph.
        """
        return list(self.__graph.edges)

    def labels(self):
        """
        Returns all node labels.
        """
        l = set()
        for node in self.nodes():
            if 'label' in node[1]:
                l.add(node[1]['label'])
        return l

    def add_node(self, id, label=""):
        """
        Adds a new node into the graph.

        Parameters
        ----------
        id : ?
            The node's identifier.
        label : ?
            The node's label (class).
        """
        self.__graph.add_node(id, label=label)

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
        self.__graph.add_edge(node1, node2)

    def induce_by_random_nodes(self, p):
        """
        Generates a graph induced by p*|N| randomly selected nodes.

        Parameters
        ----------
        p : float
            A value in the range (0,1]. Determines the size of the resulting graph's nodes set.
            The resulting graph's nodes set will have p*|N| randomly selected nodes.

        Returns the induced graph.
        """
        if p <= 0 or p > 1:
            raise Exception('p must be a value in the range (0,1]. The value of p was: {}'.format(p))
        g = Graph(self)
        rnd_nodes = random.sample(self.__graph.nodes, k=int(round((1-p)*len(self.__graph.nodes))))
        g.__graph.remove_nodes_from(rnd_nodes)
        return g

    def induce_by_random_edges(self, p):
        """
        Generates a graph induced by p*|E| randomly selected edges.

        Parameters
        ----------
        p : float
            A value in the range (0,1]. Determines the size of the resulting graph's edges set.
            The resulting graph's edges set will have p*|E| randomly selected edges.

        Returns the induced graph.
        """
        if p <= 0 or p > 1:
            raise Exception('p must be a value in the range (0,1]. The value of p was: {}'.format(p))
        g = Graph(self)
        rnd_edges = random.sample(self.__graph.edges, k=int(round((1-p)*len(self.__graph.edges))))
        g.__graph.remove_edges_from(rnd_edges)
        return g

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

    def to_networkx(self):
        return self.__graph

    def to_dgl(self):
        nodes = self.nodes()
        if len(nodes) == 0:
            return dgl.DGLGraph()
        node_attrs = []
        if 'label' in nodes[0][1]:
            node_attrs.append('label')
        return dgl.from_networkx(self.__graph, node_attrs=node_attrs)