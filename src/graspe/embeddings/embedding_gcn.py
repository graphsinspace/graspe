import itertools

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from embeddings.base.embedding import Embedding
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


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
        hub_aware=False,
        hub_fn='identity',
        badness_aware=False,
        badness_alpha=1,
        train=0.8,
        val=0.1,
        test=0.1,
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
            k-value param for LID
        hub_aware: bool
            Whether to take into account hubness of nodes
        hub_fn: str
            Which function to be used on hubness of nodes, support for identity, inverse, log, log_inverse
        badness_aware: bool
            Whether to take into account goodness of nodes (take a look at method get_goodness in graph.py)
        badness_alpha: int
            Strength of influence of goodness aware
        train : float
            Percentage of data to be used for training.
        val : float
            Percentage of data to be used for validation.
        test : float
            Percentage of data to be used for testing.
        """
        super().__init__(g, d)
        self.epochs = epochs
        self.lr = lr
        self.layer_configuration = layer_configuration
        self.act_fn = act_fn
        self.lid_aware = lid_aware
        self.lid_k = lid_k
        self.hub_aware = hub_aware
        self.hub_fn = hub_fn
        self.badness_aware = badness_aware
        self.badness_alpha = badness_alpha
        self.dgl_g = self._g.to_dgl()
        self.train = train
        self.val = val
        self.test = test

        if (self.dgl_g.in_degrees() == 0).any():
            self.dgl_g = dgl.add_self_loop(self.dgl_g)

        if deterministic:  # not thread-safe, beware if running multiple at once
            if torch.__version__ == '1.10':
                torch.use_deterministic_algorithms(True)  # Torch 1.10
            else:
                torch.set_deterministic(True)
            torch.manual_seed(0)
            np.random.seed(0)
        else:
            if torch.__version__ == '1.10':
                torch.use_deterministic_algorithms(False)  # Torch 1.10
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
        inputs = e.weight

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

        labeled_nodes = []
        labels = []
        for node in nodes:
            if "label" in node[1]:
                labeled_nodes.append(node[0])
                labels.append(node[1]["label"])
        labels = torch.tensor(labels).to(device)

        train_nid, test_nid = train_test_split(range(len(labels)), test_size=0.2, random_state=1)
        train_nid, test_nid = torch.Tensor(train_nid).long(), torch.Tensor(test_nid).long()

        optimizer = torch.optim.Adam(
            itertools.chain(net.parameters(), e.parameters()), lr=self.lr
        )

        if self.hub_aware:
            hubness_vector = torch.Tensor(list(self._g.get_hubness().values())) + 1e-5
            if self.hub_fn == 'identity':
                pass
            elif self.hub_fn == 'inverse':
                hubness_vector = 1 / hubness_vector
            elif self.hub_fn == 'log':
                hubness_vector = torch.log(hubness_vector)
            elif self.hub_fn == 'log_inverse':
                hubness_vector = 1 / torch.log(hubness_vector)
            else:
                raise Exception('{} is not supported as a hub_fn parameter. Currently implemented options are '
                                'identity, inverse, log, and log_inverse'.format(self.hub_fn))
            hubness_vector = hubness_vector / hubness_vector.norm()

        if self.badness_aware:
            badness_vector = (torch.Tensor(self._g.get_badness()) + 1) ** self.badness_alpha

        for epoch in range(1, self.epochs + 1):
            logits = net(dgl_g.to(device), inputs.to(device)).to(device)
            logp = F.log_softmax(logits, 1)

            # Re-weighting the loss with hubness and/or badness vector
            if self.hub_aware and not self.badness_aware:
                criterion_1 = F.nll_loss(logp[train_nid], labels[train_nid], reduction='none')
                criterion_1 = torch.dot(criterion_1, hubness_vector[train_nid])
            elif not self.hub_aware and self.badness_aware:
                criterion_1 = F.nll_loss(logp[train_nid], labels[train_nid], reduction='none')
                criterion_1 = torch.dot(criterion_1, badness_vector[train_nid])
            elif self.hub_aware and self.badness_aware:
                criterion_1 = F.nll_loss(logp[train_nid], labels[train_nid], reduction='none')
                criterion_1 = torch.dot(criterion_1, (hubness_vector[train_nid] + badness_vector[train_nid]) / 2)
            else:
                criterion_1 = F.nll_loss(logp[train_nid], labels[train_nid])

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

        self._embedding = self.compute_embedding(dgl_g, nodes)
        acc = self._evaluate(net, dgl_g, inputs, labels, test_nid, full=True)

    def compute_embedding(self, dgl_g, nodes):
        embedding = {}
        for i in range(len(nodes)):
            embedding[nodes[i][0]] = np.array(
                [x.item() for x in dgl_g.ndata["feat"][i]]
            )
        return embedding

    def _evaluate(self, net, dgl_g, inputs, labels, labeled_nodes, full=False):
        net.eval()
        with torch.no_grad():
            logits = net(dgl_g, inputs)
            logits = logits[labeled_nodes]
            labels = labels[labeled_nodes]
            _, indices = torch.max(logits, dim=1)
        net.train()
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
