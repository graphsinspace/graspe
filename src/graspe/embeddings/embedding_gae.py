import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as tg
import pickle
from abc import abstractmethod
from torch.nn import Sequential
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.utils import train_test_split_edges
from embeddings.base.embedding import Embedding
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch, NCLIDEstimator
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn.inits import reset


EPS = 1e-15
MAX_LOGSTD = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None,
                   hub_vector=None, hub_aware=False, nclid_vector=None,
                   nclid_aware=False, combine_fn='add'):

        # Recon loss is comprised of positive loss and negative loss.
        # Positive loss
        if hub_aware:  # Hub Aware block
            hub_vector_pos = hub_vector[pos_edge_index]
            if combine_fn == 'add':
                hub_vector_pos = hub_vector_pos.sum(axis=0)
            elif combine_fn == 'mult':
                hub_vector_pos = hub_vector_pos[0] * hub_vector_pos[1]
            else:
                raise Exception('{} is not supported as a hub_combine parameter. Currently implemented options are '
                                'add and mult'.format(combine_fn))
            hub_vector_pos /= hub_vector_pos.norm()
            pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS)
            pos_loss = torch.mul(pos_loss, hub_vector_pos).mean()

        elif nclid_aware:  # NCLID Aware block
            nclid_vector_pos = nclid_vector[pos_edge_index]
            if combine_fn == 'add':
                nclid_vector_pos = nclid_vector_pos.sum(axis=0)
            elif combine_fn == 'mult':
                nclid_vector_pos = nclid_vector_pos[0] * nclid_vector_pos[1]
            else:
                raise Exception('{} is not supported as a hub_combine parameter. Currently implemented options are '
                                'add and mult'.format(combine_fn))
            nclid_vector_pos /= nclid_vector_pos.norm()
            pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
            pos_loss = torch.mul(pos_loss, nclid_vector_pos).mean()
        else:
            pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Negative loss
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        if hub_aware:  # Hub Aware block
            hub_vector_neg = hub_vector[neg_edge_index]
            if combine_fn == 'add':
                hub_vector_neg = hub_vector_neg.sum(axis=0)
            elif combine_fn == 'mult':
                hub_vector_neg = hub_vector_neg[0] * hub_vector_neg[1]
            else:
                raise Exception('{} is not supported as a hub_combine parameter. Currently implemented options are '
                                'add and mult'.format(combine_fn))
            hub_vector_neg /= hub_vector_neg.norm()
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS)
            neg_loss = torch.mul(neg_loss, hub_vector_neg).mean()
        elif nclid_aware:  # NCLID Aware block
            nclid_vector_neg = nclid_vector[neg_edge_index]
            if combine_fn == 'add':
                nclid_vector_neg = nclid_vector_neg.sum(axis=0)
            elif combine_fn == 'mult':
                nclid_vector_neg = nclid_vector_neg[0] * nclid_vector_neg[1]
            else:
                raise Exception('{} is not supported as a hub_combine parameter. Currently implemented options are '
                                'add and mult'.format(combine_fn))
            nclid_vector_neg /= nclid_vector_neg.norm()
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS)
            neg_loss = torch.mul(neg_loss, nclid_vector_neg).mean()
        else:
            neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    def __init__(self, encoder, decoder=None):
        super(VGAE, self).__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, channel_configuration=(8,), act_fn=torch.relu):
        super(GCNEncoder, self).__init__()
        self.act_fn = act_fn
        self.input = GCNConv(in_channels, channel_configuration[0], cached=True)
        self.hidden = []
        last_num_channels = channel_configuration[0]

        for num_channels in channel_configuration[1:]:
            layer = GCNConv(last_num_channels, num_channels, cached=True)
            self.hidden.append(layer)
            last_num_channels = num_channels

        self.hidden = Sequential(*self.hidden)  # Module registration
        self.output = GCNConv(last_num_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.act_fn(self.input(x, edge_index))

        for layer in self.hidden:
            x = self.act_fn(layer(x, edge_index))

        return self.output(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, channel_configuration=(8,), act_fn=torch.relu):
        super(VariationalGCNEncoder, self).__init__()
        self.act_fn = act_fn
        self.input = GCNConv(in_channels, channel_configuration[0], cached=True)
        self.hidden = []
        last_num_channels = channel_configuration[0]

        for num_channels in channel_configuration[1:]:
            layer = GCNConv(last_num_channels, num_channels, cached=True)
            self.hidden.append(layer)
            last_num_channels = num_channels

        self.hidden = Sequential(*self.hidden)  # Module registration

        self.output_mu = GCNConv(last_num_channels, out_channels, cached=True)
        self.output_logstd = GCNConv(last_num_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.input(x, edge_index).relu()

        for layer in self.hidden:
            x = self.act_fn(layer(x, edge_index))

        return self.output_mu(x, edge_index), self.output_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GAEEmbeddingBase(Embedding):
    """
    Embedding with Graph Auto Encoders (pytorch_geometric impl.)
    Support for:
    Non-variational non-linear autoencoders
    Variational non-linear autoencoders
    Non-variational linear autoencoders
    Variational linear autoencoders
    Papers:
    https://arxiv.org/abs/1611.07308
    """

    def __init__(
        self,
        g,
        d,
        epochs=500,
        variational=False,
        linear=False,
        deterministic=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
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
        variational : bool
            Whether to use Variational autoencoders (VAEs)
        linear : bool
            Whether to use Linear Encoders for the autoencoder model
        deterministic : bool
            Whether to try and run in deterministic mode
        lr : float
            Learning rate for the optimizer
        layer_configuration : tuple[int]
            Hidden layer configuration, tuple length is depth, values are hidden sizes
        act_fn : str
            Activation function to be used, support for relu, tanh, sigmoid
        """
        super().__init__(g, d)
        self.epochs = epochs
        self.variational = variational
        self.linear = linear
        self.lr = lr
        self.layer_configuration = layer_configuration
        self.act_fn = act_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if deterministic:  # not thread-safe, beware if running multiple at once
            torch.use_deterministic_algorithms(False)  # Torch 1.10
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            torch.cuda.manual_seed_all(0)
            os.environ["PYTHONHASHSEED"] = str(0)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(False)  # Torch 1.10

    @abstractmethod
    def calculate_loss(self, z, model, train_pos_edge_index, data, x):
        pass

    def _train(self, model, optimizer, train_pos_edge_index, data, x):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        criterion_1 = self.calculate_loss(z, model, train_pos_edge_index, data, x)
        if self.variational:
            criterion_1 = criterion_1 + (1 / data.num_nodes) * model.kl_loss()
        loss = criterion_1
        loss.backward()
        optimizer.step()
        return float(loss)

    def _test(self, pos_edge_index, neg_edge_index, model, x, train_pos_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

    def embed(self):
        super().embed()

        # TODO: this seems like extra work to me, since pytorch_geometric already has
        # TODO: datasets ready to use...
        digraph = self._g.to_dgl().to_networkx()
        print(type(digraph))
        print(digraph)
        data = tg.utils.from_networkx(digraph)

        out_channels = self._d
        num_nodes = data.num_nodes
        num_features = data.num_features if data.num_features else num_nodes

        data.train_mask = data.val_mask = data.test_mask = data.y = data.label = None

        # Nemanja: pytorch_geometric needs one hot encodings of nodes as torch.tensors
        data.x = F.one_hot(torch.arange(0, data.num_nodes) % data.num_nodes).float()
        data = train_test_split_edges(data)

        if self.act_fn == "relu":
            self.act_fn = torch.relu
        elif self.act_fn == "tanh":
            self.act_fn = torch.tanh
        elif self.act_fn == "sigmoid":
            self.act_fn = torch.sigmoid

        if self.variational:
            if self.linear:
                encoder = VariationalLinearEncoder(num_features, out_channels)
            else:
                encoder = VariationalGCNEncoder(
                    num_features,
                    out_channels,
                    channel_configuration=self.layer_configuration,
                    act_fn=torch.relu,
                )
            model = VGAE(encoder)
        else:
            if self.linear:
                encoder = LinearEncoder(num_features, out_channels)
            else:
                encoder = GCNEncoder(
                    num_features,
                    out_channels,
                    channel_configuration=self.layer_configuration,
                    act_fn=torch.relu,
                )
            model = GAE(encoder)

        model = model.to(self.device)
        x = data.x.to(self.device)
        train_pos_edge_index = data.train_pos_edge_index.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        loss = -1
        for epoch in range(1, self.epochs + 1):
            loss = self._train(model, optimizer, train_pos_edge_index, data, x)
            # auc, ap = self._test(
            #     data.test_pos_edge_index,
            #     data.test_neg_edge_index,
            #     model,
            #     x,
            #     train_pos_edge_index,
            # )
            # print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}".format(epoch, auc, ap))
        # print("loss:", loss)

        with torch.no_grad():
            self._embedding = {}
            encoded = model.encode(x, train_pos_edge_index).cpu().numpy()
            for i in range(num_nodes):
                self._embedding[i] = encoded[i, :]

    def requires_labels(self):
        return False


class GAEEmbedding(GAEEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs=500,
        variational=False,
        linear=False,
        deterministic=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
    ):
        super().__init__(g, d, epochs, deterministic, variational, linear, lr, layer_configuration, act_fn)

    def calculate_loss(self, z, model, train_pos_edge_index, data, x):
        return model.recon_loss(z, train_pos_edge_index)


class GAEEmbeddingLIDAware(GAEEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs=500,
        variational=False,
        linear=False,
        deterministic=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
        lid_k=20
    ):
        super().__init__(g, d, epochs, deterministic, variational, linear, lr, layer_configuration, act_fn)
        self.lid_k = lid_k

    def calculate_loss(self, z, model, train_pos_edge_index, data, x):
        loss_1 = model.recon_loss(z, train_pos_edge_index)
        emb = {}
        encoded = model.encode(x, train_pos_edge_index).cpu().detach().numpy()
        for i in range(data.num_nodes):
            emb[i] = encoded[i, :]
        tlid = EmbLIDMLEEstimatorTorch(self._g, emb, self.lid_k)
        tlid.estimate_lids()
        total_lid = tlid.get_total_lid()
        loss_2 = F.mse_loss(total_lid, torch.tensor(self.lid_k, dtype=torch.float))
        loss = loss_1 + loss_2
        return loss


class GAEEmbeddingHubAware(GAEEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs=500,
        variational=False,
        linear=False,
        deterministic=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
        hub_fn='identity',
        hub_combine='add'
    ):
        super().__init__(g, d, epochs, deterministic, variational, linear, lr, layer_configuration, act_fn)
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
        self.hub_combine = hub_combine

    def calculate_loss(self, z, model, train_pos_edge_index, data, x):
        loss = model.recon_loss(
            z,
            train_pos_edge_index,
            hub_vector=self.hubness_vector,
            hub_aware=True,
            combine_fn=self.hub_combine)
        return loss


class GAEEmbeddingNCLIDAware(GAEEmbeddingBase):
    def __init__(
        self,
        g,
        d,
        epochs=500,
        variational=False,
        linear=False,
        deterministic=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
        nclid_combine='add',
        alpha=1,
        dataset_name=None
    ):
        super().__init__(g, d, epochs, deterministic, variational, linear, lr, layer_configuration, act_fn)
        self.nclid_combine = nclid_combine
        path = f'/home/stamenkovicd/nclids/{dataset_name}_nclids_tensor.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as file:
                nclids = pickle.load(file)
                self.nclids = nclids.to(DEVICE)
        else:
            nclid = NCLIDEstimator(g, alpha=alpha)
            nclid.estimate_lids()
            self.nclids = torch.Tensor([nclid.get_lid(node[0]) for node in self._g.nodes()]).to(DEVICE)

    def calculate_loss(self, z, model, train_pos_edge_index, data, x):
        loss = model.recon_loss(
            z,
            train_pos_edge_index,
            nclid_aware=True,
            nclid_vector=self.nclids,
            combine_fn=self.nclid_combine)
        return loss
