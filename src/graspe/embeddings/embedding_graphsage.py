import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

from embeddings.base.embedding import Embedding

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, num_classes, aggregator_type, configuration=(128, ), act_fn=torch.relu, dropout=0.0):
        super(GraphSAGE, self).__init__()
        self.hidden = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.act_fn = act_fn
        last_hidden_size = configuration[0]

        self.input = SAGEConv(in_feats, configuration[0], aggregator_type).to(device)

        for layer_size in configuration[1:]:
            layer = SAGEConv(last_hidden_size, layer_size, aggregator_type).to(device)
            self.hidden.append(layer)
            last_hidden_size = layer_size

        self.hidden = nn.Sequential(*self.hidden)  # Module registration

        self.output = SAGEConv(last_hidden_size, configuration[0], aggregator_type).to(device)

        # idea for embedding extraction from: https://github.com/stellargraph/stellargraph/issues/1586
        self.fc = nn.Linear(configuration[0], num_classes)

    def forward(self, g, inputs):
        h = self.dropout(inputs)
        h = self.act_fn(self.input(g, h))

        for layer in self.hidden:
            h = self.dropout(self.act_fn(layer(g, h)))

        h = self.output(g, h)
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
        dropout=0.0,
        layer_configuration=(128, ),
        act_fn="relu",
        train=0.8,
        val=0.1,
        test=0.1,
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
        dropout : float
            Probability of applying a dropout in hidden layers
        layer_configuration : tuple[int]
            Hidden layer configuration, tuple length is depth, values are hidden sizes
        act_fn : str
            Activation function to be used, support for relu, tanh, sigmoid
        train : float
            Percentage of data to be used for training.
        val : float
            Percentage of data to be used for validation.
        test : float
            Percentage of data to be used for testing.
        lr : float
            Learning rate
        deterministic : bool
            Whether to try and run in deterministic mode
        """
        super().__init__(g, d)
        self._epochs = epochs
        self.layer_configuration = layer_configuration
        self.act_fn = act_fn
        self.train = train
        self.val = val
        self.test = test
        self.dropout = dropout
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

        if self.act_fn == "relu":
            self.act_fn = torch.relu
        elif self.act_fn == "tanh":
            self.act_fn = torch.tanh
        elif self.act_fn == "sigmoid":
            self.act_fn = torch.sigmoid

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
            n_classes,
            "gcn",
            configuration=self.layer_configuration,
            act_fn=torch.relu,
            dropout=self.dropout,
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
