import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from embeddings.base.embedding import Embedding
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        num_classes,
        aggregator_type,
        configuration=(128,),
        act_fn=torch.relu,
        dropout=0.0,
    ):
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
        h = self.act_fn(self.input(g, inputs))

        for layer in self.hidden:
            h = self.dropout(self.act_fn(layer(g, h)))

        h = self.output(g, h)
        return self.fc(h), h


class GraphSageEmbeddingBase(Embedding):
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
        layer_configuration=(128,),
        act_fn="relu",
        train=0.8,
        val=0.1,
        test=0.1,
        lr=1e-2,
        deterministic=False,
        verbose=True,
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
        verbose : boolean
            Whether to output train data
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
        self.verbose = verbose
        if deterministic:  # not thread-safe, beware if running multiple at once
            torch.use_deterministic_algorithms(True)  # Torch 1.10
            # torch.set_deterministic(True)
            torch.manual_seed(0)
            np.random.seed(0)
        else:
            torch.use_deterministic_algorithms(False)  # Torch 1.10
            # torch.set_deterministic(False)

    @abstractmethod
    def compute_loss(self, logits, labels, train_nid):
        pass

    def embed(self):
        super().embed()

        if self.act_fn == "relu":
            self.act_fn = torch.relu
        elif self.act_fn == "tanh":
            self.act_fn = torch.tanh
        elif self.act_fn == "sigmoid":
            self.act_fn = torch.sigmoid

        g = self._g.to_dgl()
        nodes = self._g.nodes()

        graph_adj = self._g.to_adj_matrix().toarray()
        inputs = torch.Tensor(graph_adj)
        in_feats = graph_adj.shape[0]
        # inputs = nn.Embedding(num_nodes, self._d).to(device)

        labeled_nodes = []
        labels = []
        for node in nodes:
            if "label" in node[1]:
                labeled_nodes.append(node[0])
                labels.append(node[1]["label"])
        labels = torch.tensor(labels).to(device)

        train_nid, test_nid = train_test_split(range(len(labels)), test_size=0.2, random_state=1)
        train_nid, test_nid = torch.Tensor(train_nid).long(), torch.Tensor(test_nid).long()

        n_classes = len(set(labels.numpy()))

        # graph preprocess and calculate normalization factor
        g = dgl.remove_self_loop(g)

        # create GraphSAGE model
        net = GraphSAGE(
            in_feats,
            n_classes,
            "gcn",
            configuration=self.layer_configuration,
            act_fn=self.act_fn,
            dropout=self.dropout,
        )

        # use optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=5e-4)

        for epoch in range(self._epochs):
            net.train()
            if epoch >= 3:
                t0 = time.time()
            logits, _ = net(g, inputs)

            loss = self.compute_loss(logits, labels, train_nid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = self._evaluate(net, g, inputs, labels, test_nid, full=True)
        if self.verbose:
            print("Test Accuracy {:.4f}".format(acc))

        with torch.no_grad():
            _, embedding = net(g, inputs)
            self._embedding = {}

            for i in range(len(embedding)):
                self._embedding[i] = embedding[i].numpy()

    def _evaluate(self, net, dgl_g, inputs, labels, labeled_nodes, full=False):
        net.eval()
        with torch.no_grad():
            logits, _ = net(dgl_g, inputs)
            logits = logits[labeled_nodes]
            labels = labels[labeled_nodes]
            _, indices = torch.max(logits, dim=1)
        net.train()
        indices, labels = indices.cpu().numpy(), labels.cpu().numpy()
        accuracy = accuracy_score(indices, labels)

        if full:
            precision = precision_score(indices, labels, average='macro')
            recall = recall_score(indices, labels, average='macro')
            f1 = (2 * precision * recall) / (precision + recall)
            print('Accuracy = {:.4f}'.format(accuracy))
            print('Precision = {:.4f}'.format(precision))
            print('Recall = {:.4f}'.format(recall))
            print('F1 Score = {:.4f}'.format(f1))
            print('Confusion Matrix \n', confusion_matrix(indices, labels))
        return accuracy

    def requires_labels(self):
        return True


class GraphSageEmbedding(GraphSageEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs,
        dropout=0.0,
        layer_configuration=(128,),
        act_fn="relu",
        train=0.8,
        val=0.1,
        test=0.1,
        lr=1e-2,
        deterministic=False,
    ):
        super().__init__(g, d, epochs, dropout, deterministic, lr, layer_configuration, act_fn, train, val, test)

    def compute_loss(self, logits, labels, train_nid):
        return F.cross_entropy(logits[train_nid], labels[train_nid])


class GraphSageEmbeddingLIDAware(GraphSageEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs,
        dropout=0.0,
        layer_configuration=(128,),
        act_fn="relu",
        train=0.8,
        val=0.1,
        test=0.1,
        lr=1e-2,
        deterministic=False,
        lid_k=20
    ):
        super().__init__(g, d, epochs, dropout, deterministic, lr, layer_configuration, act_fn, train, val, test)
        self.lid_k = lid_k

    def compute_loss(self, logits, labels, train_nid):
        loss_1 = F.cross_entropy(logits[train_nid], labels[train_nid])

        _, embedding = net(g, inputs)
        emb = {}
        for i in range(len(embedding)):
            emb[i] = embedding[i].detach().numpy()
        tlid = EmbLIDMLEEstimatorTorch(self._g, emb, self.lid_k)
        tlid.estimate_lids()
        total_lid = tlid.get_total_lid()
        loss_2 = F.mse_loss(total_lid, torch.tensor(self.lid_k, dtype=torch.float))
        loss = loss_1 + loss_2
        return loss


class GraphSageEmbeddingHubAware(GraphSageEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs,
        dropout=0.0,
        layer_configuration=(128,),
        act_fn="relu",
        train=0.8,
        val=0.1,
        test=0.1,
        lr=1e-2,
        deterministic=False,
        hub_fn='identity',
    ):
        super().__init__(g, d, epochs, dropout, deterministic, lr, layer_configuration, act_fn, train, val, test)
        hubness_vector = torch.Tensor(list(self._g.get_hubness().values())) + 1e-5
        if hub_fn == 'identity':
            pass
        elif hub_fn == 'inverse':
            hubness_vector = 1 / hubness_vector
        elif hub_fn == 'log':
            hubness_vector = torch.log(hubness_vector)
        elif hub_fn == 'log_inverse':
            hubness_vector = 1 / torch.log(hubness_vector)
        else:
            raise Exception('{} is not supported as a hub_fn parameter. Currently implemented options are '
                            'identity, inverse, log, and log_inverse'.format(hub_fn))
        hubness_vector = hubness_vector / hubness_vector.norm()
        self.hubness_vector = hubness_vector.to(device)

    def compute_loss(self, logits, labels, train_nid):
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])
        loss = torch.dot(loss, self.hubness_vector[train_nid])
        return loss


class GraphSageEmbeddingBadAware(GraphSageEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs,
        dropout=0.0,
        layer_configuration=(128,),
        act_fn="relu",
        train=0.8,
        val=0.1,
        test=0.1,
        lr=1e-2,
        deterministic=False,
        badness_alpha=1,
    ):
        super().__init__(g, d, epochs, dropout, deterministic, lr, layer_configuration, act_fn, train, val, test)
        badness_vector = (torch.Tensor(self._g.get_badness()) + 1) ** badness_alpha
        badness_vector = badness_vector / badness_vector.norm()
        self.badness_vector = badness_vector.to(device)

    def compute_loss(self, logits, labels, train_nid):
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])
        loss = torch.dot(loss, self.badness_vector[train_nid])
        return loss
