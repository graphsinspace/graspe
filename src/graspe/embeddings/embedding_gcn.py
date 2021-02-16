import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import dgl

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

    def __init__(self, g, d, epochs, deterministic=False, add_self_loop=False):
        """
        Parameters
        ----------
        g : common.graph.Graph
            The original graph.
        d : int
            Dimensionality of the embedding.
        epochs : int
            Number of epochs.
        deterministic : bool
            Whether to try and run in deterministic mode
        add_self_loop : bool
            Whether to add self loop to the DGL dataset
        """
        super().__init__(g, d)
        self._epochs = epochs
        self.add_self_loop = add_self_loop
        if deterministic:  # not thread-safe, beware if running multiple at once
            torch.set_deterministic(True)
            torch.manual_seed(0)
            np.random.seed(0)
        else:
            torch.set_deterministic(False)

    def embed(self):
        super().embed()

        nodes = self._g.nodes()
        num_nodes = len(nodes)
        labels = self._g.labels()

        dgl_g = self._g.to_dgl()

        if self.add_self_loop:
            dgl_g = dgl.add_self_loop(dgl_g)

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
        # print("loss: %.4f" % loss.item())

        self._embedding = {}
        for i in range(len(nodes)):
            self._embedding[nodes[i][0]] = np.array(
                [x.item() for x in dgl_g.ndata["feat"][i]]
            )

    def requires_labels(self):
        return True
