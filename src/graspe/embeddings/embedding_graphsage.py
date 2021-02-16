import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import dgl
from dgl.data import load_data
from dgl.data.citation_graph import load_cora

from common.dataset_pool import DatasetPool
from embeddings.base.embedding import Embedding
from dgl.nn.pytorch import GraphConv, SAGEConv


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout,
        aggregator_type,
        fc_dim,
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, fc_dim, aggregator_type)
        )  # activation None
        # idea for embedding extraction from: https://github.com/stellargraph/stellargraph/issues/1586
        self.fc = nn.Linear(fc_dim, n_classes)

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return self.fc(h), h


class GraphSageEmbedding(Embedding):
    """
    Embedding with GraphSage (DGL/PyTorch).

    https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage

    Members:

    - g : graph to compute embedding for
    - d : dimension of the input and hidden embedding

    """

    def __init__(
        self,
        g,
        d,
        epochs,
        train=0.8,
        val=0.1,
        test=0.1,
        hidden=16,
        layers=1,
        lr=1e-2,
        deterministic=False,
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
        train : float
            Percentage of data to be used for training.
        val : float
            Percentage of data to be used for validation.
        test : float
            Percentage of data to be used for testing.
        hidden : int
            Number of hidden neurons in a SageConv layer
        layers : int
            Number of SageConv layers
        lr : float
            Learning rate
        deterministic : bool
            Whether to try and run in deterministic mode
        """
        super().__init__(g, d)
        self._epochs = epochs
        self.train = train
        self.val = val
        self.test = test
        self.hidden = hidden
        self.layers = layers
        self.lr = lr
        if deterministic:  # not thread-safe, beware if running multiple at once
            torch.set_deterministic(True)
            torch.manual_seed(0)
            np.random.seed(0)
        else:
            torch.set_deterministic(False)

    def _evaluate(self, model, graph, features, labels, nid):
        model.eval()
        with torch.no_grad():
            logits, _ = model(graph, features)
            logits = logits[nid]
            labels = labels[nid]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def embed(self):
        super().embed()

        g = self._g.to_dgl()

        num_nodes = len(g)

        train_num = int(num_nodes * self.train)
        val_num = int(num_nodes * self.val)
        test_num = int(num_nodes * self.test)

        # not using attrs, using node degrees as features
        degrees = [self._g.to_networkx().degree(i) for i in range(self._g.nodes_cnt())]
        max_degree = max(degrees)
        features = torch.zeros((num_nodes, max_degree + 1))

        for i, d in enumerate(degrees):
            features[i][d] = 1  # one hot vector, node degree

        labels = g.ndata["label"]
        train_mask = torch.tensor(
            [True] * train_num + [False] * test_num + [False] * val_num
        )
        val_mask = torch.tensor(
            [True] * train_num + [True] * test_num + [False] * val_num
        )
        test_mask = torch.tensor(
            [True] * train_num + [False] * test_num + [True] * val_num
        )
        in_feats = features.shape[1]
        n_classes = len(set(labels.numpy()))

        train_nid = train_mask.nonzero().squeeze()
        val_nid = val_mask.nonzero().squeeze()
        test_nid = test_mask.nonzero().squeeze()

        # graph preprocess and calculate normalization factor
        g = dgl.remove_self_loop(g)
        n_edges = g.number_of_edges()

        # create GraphSAGE model
        model = GraphSAGE(
            in_feats,
            self.hidden,
            n_classes,
            self.layers,
            F.relu,
            0.5,
            "gcn",
            fc_dim=self._d,
        )

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)

        # initialize graph
        dur = []
        for epoch in range(self._epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits, _ = model(g, features)
            loss = F.cross_entropy(logits[train_nid], labels[train_nid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = self._evaluate(model, g, features, labels, val_nid)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000
                )
            )

        acc = self._evaluate(model, g, features, labels, test_nid)
        print("Test Accuracy {:.4f}".format(acc))

        with torch.no_grad():
            _, embedding = model(g, features)
            self._embedding = {}

            for i in range(len(embedding)):
                self._embedding[i] = embedding[i].numpy()

    def requires_labels(self):
        return True
