import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np

from common.graph import Graph
from embeddings.base.embedding import Embedding
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    """
    Example Graph-Convolutional neural network implementation. Single hidden layer.
    """

    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h


class GCNEmbedding(Embedding):
    """
    Embedding with Graph Convolutional Neural Networks (DGL/PyTorch).

    Members:

    - g : graph to compute embedding for
    - d : dimension of the input and hidden embedding

    Necessarry args:

    - epochs : number of epochs for training (int)
    - labeles_nodes : torch.tensor with id's of nodes with labels
    - labels : labels for labeled_nodes (torch.tensor)
    """

    def __init__(self, g, d, epochs):
        """
        Parameters
        ----------
        g : common.graph.Graph
            The original graph.
        d : int
            Dimensionality of the embedding.
        epochs : int
            Number of epochs.
        """
        super().__init__(g, d)
        self._epochs = epochs

    def embed(self):
        nodes = self._g.nodes()
        num_nodes = len(nodes)
        labels = self._g.labels()

        dgl_g = self._g.to_dgl()
        e = nn.Embedding(num_nodes, self._d)
        dgl_g.ndata["feat"] = e.weight
        net = GCN(self._d, self._d, len(labels))

        inputs = e.weight
        labeled_nodes = []
        labels = []
        for node in nodes:
            if "label" in node[1]:
                labeled_nodes.append(node[0])
                labels.append(node[1]["label"])
        labels = torch.tensor(labels)

        optimizer = torch.optim.Adam(
            itertools.chain(net.parameters(), e.parameters()), lr=0.01
        )
        for epoch in range(self._epochs):
            logits = net(dgl_g, inputs)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[labeled_nodes], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        #print("loss: %.4f" % loss.item())

        self._embedding = {}
        for i in range(len(nodes)):
            self._embedding[nodes[i][0]] = np.array(
                [x.item() for x in dgl_g.ndata["feat"][i]]
            )
