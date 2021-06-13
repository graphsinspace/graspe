import itertools

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from embeddings.base.embedding import Embedding
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class GCN(nn.Module):
    """
    Example Graph-Convolutional neural network implementation. Single or multiple hidden layer.
    """

    def __init__(self, in_feats, num_classes, configuration=(128,), act_fn=torch.relu):
        super(GCN, self).__init__()
        self.act_fn = act_fn
        self.input = GraphConv(in_feats, configuration[0])
        self.hidden = []
        last_hidden_size = configuration[0]

        for layer_size in configuration[1:]:
            layer = GraphConv(last_hidden_size, layer_size).to(device)
            self.hidden.append(layer)
            last_hidden_size = layer_size

        self.hidden = nn.Sequential(*self.hidden)  # Module registration

        self.output = GraphConv(last_hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.act_fn(self.input(g, inputs))

        for layer in self.hidden:
            h = self.act_fn(layer(g, h))

        h = self.output(g, h)
        return h


class GCNEmbedding(Embedding):
    """
    Embedding with Graph Convolutional Neural Networks (DGL/PyTorch).

    Members:

    - g : graph to compute embedding for
    - d : dimension of the input and hidden embedding

    Necessarry args:

    - epochs : number of epochs for training (int)
    - n_layers : number of hidden layer in the GCN
    - dropout : probability of a dropout in hidden layers of the GCN
    - labeled_nodes : torch.tensor with id's of nodes with labels
    - labels : labels for labeled_nodes (torch.tensor)
    """

    def __init__(
        self,
        g,
        d,
        epochs,
        deterministic=False,
        lr=0.01,
        layer_configuration=(128,),
        act_fn="relu",
        lid_aware=False,
        lid_k=20,
        community_labels=False
    ):
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
        lr : float
            Learning rate for the optimizer
        layer_configuration : tuple[int]
            Hidden layer configuration, tuple length is depth, values are hidden sizes
        act_fn : str
            Activation function to be used, support for relu, tanh, sigmoid
        lid_aware : bool
            Whether to optimize for lower LID
        lid_k : int
            k-value param for LID]
        community_labels : bool
            Whether to use labels obtained through a community detection algorithm
        """
        super().__init__(g, d)
        self.epochs = epochs
        self.lr = lr
        self.layer_configuration = layer_configuration
        self.act_fn = act_fn
        self.lid_aware = lid_aware
        self.lid_k = lid_k
        if community_labels:
            self._g.set_community_labels()
        self.dgl_g = self._g.to_dgl()
        if (self.dgl_g.in_degrees() == 0).any():
            self.dgl_g = dgl.add_self_loop(self.dgl_g)

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

        dgl_g = self.dgl_g.to(device)

        e = nn.Embedding(num_nodes, self._d).to(device)
        dgl_g.ndata["feat"] = e.weight

        if self.act_fn == "relu":
            self.act_fn = torch.relu
        elif self.act_fn == "tanh":
            self.act_fn = torch.tanh
        elif self.act_fn == "sigmoid":
            self.act_fn = torch.sigmoid

        net = GCN(
            self._d,
            len(labels),
            act_fn=self.act_fn,
            configuration=self.layer_configuration,
        )

        net = net.to(device)

        inputs = e.weight
        labeled_nodes = []
        labels = []
        for node in nodes:
            if "label" in node[1]:
                labeled_nodes.append(node[0])
                labels.append(node[1]["label"])
        labels = torch.tensor(labels).to(device)
        optimizer = torch.optim.Adam(
            itertools.chain(net.parameters(), e.parameters()), lr=self.lr
        )
        for epoch in range(self.epochs):
            logits = net(dgl_g.to(device), inputs.to(device)).to(device)
            logp = F.log_softmax(logits, 1)
            criterion_1 = F.nll_loss(logp[labeled_nodes], labels)
            if self.lid_aware:
                emb = self.compute_embedding(dgl_g, nodes)
                tlid = EmbLIDMLEEstimatorTorch(self._g, emb, self.lid_k)
                tlid.estimate_lids()
                total_lid = tlid.get_total_lid()
                criterion_2 = F.mse_loss(
                    total_lid, torch.tensor(self.lid_k, dtype=torch.float)
                )
                loss = criterion_1 + criterion_2
            else:
                loss = criterion_1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        # print("loss: %.4f" % loss.item())

        self._embedding = self.compute_embedding(dgl_g, nodes)

    def compute_embedding(self, dgl_g, nodes):
        embedding = {}
        for i in range(len(nodes)):
            embedding[nodes[i][0]] = np.array(
                [x.item() for x in dgl_g.ndata["feat"][i]]
            )
        return embedding

    def requires_labels(self):
        return True
