from typing import Optional

import dgl
import torch

from common.base.graph import Graph


class DGLGraph(Graph):
    """
    A model class for graph containing implementation for DGL DGLGraphs. Inherits from common Graph.

    Attributes
    ----------
    __graph : DGLGraph
    """

    def __init__(
        self,
        u: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        num_nodes: int = None,
    ):
        """
        DGL Graph constructor

        Parameters
        ----------
        u : torch.tensor containing left-hand-sides of the edge definitions
        v : torch.tensor containing right-hand-sides of the edge definitions
        num_nodes : If the node with the largest ID is isolated (meaning no edges),
                    then one needs to explicitly set the number of nodes

        Examples
        ----------
        DGLGraph(torch.tensor([2, 5, 3]), torch.tensor([3, 5, 0])) -> creates a graph 2->3, 5->5, 3->0
        """
        super().__init__()
        if u is not None and v is not None:
            self.__graph = dgl.graph((u, v), num_nodes=num_nodes)

    @staticmethod
    def from_existing(dglg):
        g = DGLGraph(u=None, v=None, num_nodes=0)
        g.__graph = dglg
        return g

    def add_nodes(self, num, data: dict = None, ntype: str = None):
        """
        Wrapper method for adding nodes to a graph.

        Parameters
        ----------
        num: number of new nodes to add, ids are automatically added
        data: node data (dict)
        ntype: node type as 'str', optional if nodes are all the same type
        """
        self.__graph.add_nodes(num, data, ntype)

    def add_node(self, node_id, label=None, features: dict = None):
        """
        Adds a single node to the graph, id is automatically added to the
        dgl.DGLGraph structure, whilst the id paramter is saved into node data

        Parameters
        ----------
        node_id: node's unique identifier
        label: label of the node (e.g. for classification purposes)
        features: node data
        """
        self.add_nodes(
            num=1, data={"features": features, "label": label, "id": node_id}
        )

    def add_edges(self, source, target, data=None, etype=None):
        """
        Wrapper method for adding edges to a graph.

        Parameters
        ----------
        source: either id of the node or a reference, can be torch.tensor or int
        target: either id of the node or a reference, can be torch.tensor or int
        data: data of the edge
        etype: edge type as 'str', optional if edges are all the same type
        """
        self.__graph.add_edges(source, target, data, etype)

    def add_edge(self, node1, node2, weight=0.0):
        """
        Adds a single edge to the graph, between node1 and node2.

        Parameters
        ----------
        node1: either id of the node or a reference, can be torch.tensor or int
        node2: either id of the node or a reference, can be torch.tensor or int
        weight: optional, weight of the edge
        """
        self.add_edges(node1, node2, data={"weight": weight})

    def nodes(self):
        """
        Returns all the nodes of the underlying graph structure as a tensor.
        """
        return self.__graph.nodes()

    def edges(self):
        """
        Returns all the edges of the underlying graph structure as a tensor.
        """
        return self.__graph.edges()

    def nodes_edges(self):
        """
        Returns all the edges and nodes of the underlying graph structure as a dict
        of tensors.
        """
        return {"nodes": self.nodes(), "edges": self.edges()}

    def to_networkx(self):
        """
        Converts DGL graph to the NetworkX format

        Returns
        -------
        networkx.DiGraph
        """
        return self.__graph.to_networkx()

    @property
    def impl(self):
        return self.__graph

    # TODO:

    def map(self, g):
        pass

    def induce_by_random_nodes(self, p):
        pass

    def induce_by_random_edges(self, p):
        pass

    def link_precision(self, g):
        pass
