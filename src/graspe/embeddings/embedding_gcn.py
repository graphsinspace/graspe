import os
import itertools
import pickle
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from embeddings.base.embedding import Embedding
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch, NCLIDEstimator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
            layer = GraphConv(last_hidden_size, layer_size).to(DEVICE)
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


class GCNEmbeddingBase(Embedding):
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
        self.dgl_g = self._g.to_dgl()
        self.train = train
        self.val = val
        self.test = test

        if (self.dgl_g.in_degrees() == 0).any():
            self.dgl_g = dgl.add_self_loop(self.dgl_g)

        if deterministic:  # not thread-safe, beware if running multiple at once
            torch.use_deterministic_algorithms(True)  # Torch 1.10
            #torch.set_deterministic(True)
            torch.manual_seed(0)
            np.random.seed(0)
        else:
            torch.use_deterministic_algorithms(False)  # Torch 1.10
            #torch.set_deterministic(False)

    @abstractmethod
    def calculate_loss(self, logp, labels, train_nid):
        pass

    def embed(self):
        super().embed()

        nodes = self._g.nodes()
        num_nodes = len(nodes)
        labels = self._g.labels()

        dgl_g = self.dgl_g.to(DEVICE)

        e = nn.Embedding(num_nodes, self._d).to(DEVICE)
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
        net = net.to(DEVICE)

        labeled_nodes = []
        labels = []
        for node in nodes:
            if "label" in node[1]:
                labeled_nodes.append(node[0])
                labels.append(node[1]["label"])
        labels = torch.tensor(labels).to(DEVICE)

        train_nid, test_nid = train_test_split(range(len(labels)), test_size=0.2, random_state=1)
        train_nid, test_nid = torch.Tensor(train_nid).long(), torch.Tensor(test_nid).long()

        optimizer = torch.optim.Adam(itertools.chain(net.parameters(), e.parameters()), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            logits = net(dgl_g.to(DEVICE), inputs.to(DEVICE)).to(DEVICE)
            logp = F.log_softmax(logits, 1)

            loss = self.calculate_loss(logp, labels, train_nid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._embedding = self.compute_embedding(dgl_g, nodes)
        acc, prec, rec, f1 = self._evaluate(net, dgl_g, inputs, labels, test_nid, full=True)
        return acc, prec, rec, f1
    
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
        indices, labels = indices.cpu().numpy(), labels.cpu().numpy()
        accuracy = accuracy_score(indices, labels)
        if not full:
            return accuracy
        else:
            precision = precision_score(indices, labels, average='macro')
            recall = recall_score(indices, labels, average='macro')
            f1 = (2 * precision * recall) / (precision + recall)
            print('Accuracy = {:.4f}'.format(accuracy))
            print('Precision = {:.4f}'.format(precision))
            print('Recall = {:.4f}'.format(recall))
            print('F1 Score = {:.4f}'.format(f1))
            print('Confusion Matrix \n', confusion_matrix(indices, labels))
            return accuracy, precision, recall, f1
        
    def requires_labels(self):
        return True


class GCNEmbedding(GCNEmbeddingBase):
    def __init__(
            self,
            g,
            d,
            epochs,
            deterministic=False,
            lr=0.01,
            layer_configuration=(128,),
            act_fn="relu",
            train=0.8,
            val=0.1,
            test=0.1,
    ):
        super().__init__(g, d, epochs, deterministic, lr, layer_configuration, act_fn, train, val, test)

    def calculate_loss(self, logp, labels, train_nid):
        return F.nll_loss(logp[train_nid], labels[train_nid])


class GCNEmbeddingLIDAware(GCNEmbeddingBase):
    def __init__(
            self,
            g,
            d,
            epochs,
            deterministic=False,
            lr=0.01,
            layer_configuration=(128,),
            act_fn="relu",
            train=0.8,
            val=0.1,
            test=0.1,
            lid_k=20
    ):
        super().__init__(g, d, epochs, deterministic, lr, layer_configuration, act_fn, train, val, test)
        self.lid_k = lid_k

    def calculate_loss(self, logp, labels, train_nid):
        loss_1 = F.nll_loss(logp[train_nid], labels[train_nid])
        emb = self.compute_embedding(self.dgl_g.to(DEVICE), logp)
        tlid = EmbLIDMLEEstimatorTorch(self._g, emb, self.lid_k)
        tlid.estimate_lids()
        total_lid = tlid.get_total_lid()
        loss_2 = F.mse_loss(
            total_lid, torch.tensor(self.lid_k, dtype=torch.float)
        )
        loss = loss_1 + loss_2
        return loss


class GCNEmbeddingHubAware(GCNEmbeddingBase):
    def __init__(
            self,
            g,
            d,
            epochs,
            deterministic=False,
            lr=0.01,
            layer_configuration=(128,),
            act_fn="relu",
            train=0.8,
            val=0.1,
            test=0.1,
            hub_fn='identity'
    ):
        super().__init__(g, d, epochs, deterministic, lr, layer_configuration, act_fn, train, val, test)
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
        self.hubness_vector = hubness_vector.to(DEVICE)

    def calculate_loss(self, logp, labels, train_nid):
        loss = F.nll_loss(logp[train_nid], labels[train_nid])
        loss = torch.mul(loss, self.hubness_vector[train_nid]).mean()
        return loss


class GCNEmbeddingBadAware(GCNEmbeddingBase):
    def __init__(
            self,
            g,
            d,
            epochs,
            deterministic=False,
            lr=0.01,
            layer_configuration=(128,),
            act_fn="relu",
            train=0.8,
            val=0.1,
            test=0.1,
            badness_alpha=1
    ):
        super().__init__(g, d, epochs, deterministic, lr, layer_configuration, act_fn, train, val, test)
        badness_vector = (torch.Tensor(self._g.get_badness()) + 1) ** badness_alpha
        badness_vector = badness_vector / badness_vector.norm()
        self.badness_vector = badness_vector.to(DEVICE)

    def calculate_loss(self, logp, labels, train_nid):
        loss = F.nll_loss(logp[train_nid], labels[train_nid])
        loss = torch.mul(loss, self.badness_vector[train_nid]).mean()
        return loss
    
    
class GCNEmbeddingNCLID(GCNEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs,
        deterministic=False,
        lr=0.01,
        layer_configuration=(128,),
        act_fn="relu",
        train=0.8,
        val=0.1,
        test=0.1,
        alpha=1,
        dataset_name=None
    ):
        super().__init__(g, d, epochs, deterministic, lr, layer_configuration, act_fn, train, val, test)
        path = f'/home/stamenkovicd/nclids/{dataset_name}_nclids_tensor.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                nclids = pickle.load(file)
                self.nclids = nclids.to(DEVICE)
        else:
            nclid = NCLIDEstimator(g, alpha=1)
            nclid.estimate_lids()
            self.nclids = torch.Tensor([nclid.get_lid(node[0]) for node in self._g.nodes()]).to(DEVICE)

    def calculate_loss(self, logp, labels, train_nid):
        loss = F.nll_loss(logp[train_nid], labels[train_nid])
        loss = torch.mul(loss, self.nclids[train_nid]).mean()
        return loss
