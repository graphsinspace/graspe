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
        label: string
            Name of node label.
        ----------
        """
        if isinstance(data, Graph):
            self.__graph = nx.DiGraph(data.__graph)
        else:
            if isinstance(data, dgl.DGLGraph):
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

    def edges(self, node=None):
        """
        Returns edges of the graph.
        """
        return list(self.__graph.edges if node==None else self.__graph.edges[node])

    def nodes_cnt(self):
        """
        Returns number of nodes in the graph.
        """
        return len(self.__graph)

    def get_label(self, node):
        """
        Returns label for the given node
        
        Parameters
        ----------
        node : int
            Id of a node.

        If a node with the given id exists, and if that node has a label, the method returns the node's label.
        Otherwise the method returns None.
        """
        if not node in self.__graph:
            return None
        if not 'label' in self.__graph[node]:
            return None
        return self.__graph[node]['label']

    def labels(self):
        """
        Returns set of all possible node labels.
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
        id : int
            The node's identifier.
        label : int
            The node's label (class).
        """
        self.__graph.add_node(id, label=label)

    def add_edge(self, node1, node2, weight=0):
        """
        Adds a new edge into the graph.

        Parameters
        ----------
        node1 : int
            Identifier of the edge's starting node.
        node2 : int
            Identifier of the edge's ending node.
        weight: numeric
            Weight of the edge.
        """
        self.__graph.add_edge(node1, node2, w=weight)

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
        cnt = 0
        for edge in g.__graph.edges:
            if self.__graph.has_edge(*edge):
                cnt += 1

        return cnt / len(g.__graph.edges)

    def map(self, g):
        """
        MAP estimates precision for every node and computes the average over all nodes.

        Parameters
        ----------
        g : common.graph.Graph
            A graph object.
        """
        s = 0
        nodes_cnt = 0
        for node in g.__graph.nodes:
            predicted_edges = g.__graph.edges(node)
            if len(predicted_edges) == 0:
                continue
            nodes_cnt += 1
            real_edges = self.__graph.edges(node)
            node_s = 0
            for p_edge in predicted_edges:
                if p_edge in real_edges:
                    node_s += 1
            s += node_s / len(predicted_edges)
        return s / nodes_cnt

    def to_networkx(self):
        """
        Returns a networkx representation of the graph.
        """
        return self.__graph

    def to_dgl(self):
        """
        Returns a DGL representation of the graph.
        """
        nodes = self.nodes()
        if len(nodes) == 0:
            return dgl.DGLGraph()
        node_attrs = []
        if 'label' in nodes[0][1]:
            node_attrs.append('label')
        return dgl.from_networkx(self.__graph, node_attrs=node_attrs)

    def to_adj_matrix(self):
        """
        Returns adjacency matrix of the graph.
        """
        nodes = self.nodes()
        mapping = {nodes[i][0]: i for i in range(len(nodes))}
        node_size = len(nodes)
        adj = np.zeros((node_size, node_size))
        for edge in self.__graph.edges():
            adj[mapping[edge[0]]][mapping[edge[1]]] = edge['w'] if 'w' in edge else 1
        return mapping, adj