import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from common.dgl_graph import DGLGraph
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

    def embed(self, args=None):
        return self._train_embed(self.g, input_dimension=self.d, embedding_dimension=self.d,
                                 labeled_nodes=args["labeled_nodes"],
                                 labels=args["labels"], epochs=args["epochs"])

    def _train_embed(self, g: DGLGraph, input_dimension=5, embedding_dimension=5,
                     labeled_nodes=None, labels=None,
                     epochs=50):
        num_nodes = len(g.nodes())
        g = g.impl

        self.embed = nn.Embedding(num_nodes, input_dimension)
        g.ndata['feat'] = self.embed.weight

        self.net = GCN(input_dimension, embedding_dimension, len(set(labels)))

        inputs = self.embed.weight

        optimizer = torch.optim.Adam(itertools.chain(self.net.parameters(), self.embed.parameters()), lr=0.01)

        all_logits = []
        for epoch in range(epochs):
            logits = self.net(g, inputs)
            all_logits.append(logits.detach())
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[labeled_nodes], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

        print("completed training")
        self._embedding = self.embed.weight

    def __getitem__(self, node):
        return self._embedding[node]
