import torch
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges

from embeddings.base.embedding import Embedding


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


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


class GAEEmbedding(Embedding):
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

    def __init__(self, g, d, epochs=500, variational=False, linear=False):
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
        """
        super().__init__(g, d)
        self.epochs = epochs
        self.variational = variational
        self.linear = linear

    def _train(self, model, optimizer, train_pos_edge_index, data, x):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index)
        if self.variational:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)

    def _test(self, pos_edge_index, neg_edge_index, model, x, train_pos_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

    def embed(self):
        # TODO: this seems like extra work to me, since pytorch_geometric already has
        # TODO: datasets ready to use...
        data = tg.utils.from_networkx(self._g.to_networkx())

        out_channels = self._d
        num_nodes = data.num_nodes
        num_features = data.num_features if data.num_features else num_nodes

        data.train_mask = data.val_mask = data.test_mask = data.y = data.label = None

        # Nemanja: pytorch_geometric needs one hot encodings of nodes as torch.tensors
        data.x = F.one_hot(torch.arange(0, data.num_nodes) % data.num_nodes).float()
        data = train_test_split_edges(data)

        if self.variational:
            if self.linear:
                encoder = VariationalLinearEncoder(num_features, out_channels)
            else:
                encoder = VariationalGCNEncoder(num_features, out_channels)
            model = VGAE(encoder)
        else:
            if self.linear:
                encoder = LinearEncoder(num_features, out_channels)
            else:
                encoder = GCNEncoder(num_features, out_channels)
            model = GAE(encoder)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        x = data.x.to(device)
        train_pos_edge_index = data.train_pos_edge_index.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loss = -1
        for epoch in range(1, self.epochs + 1):
            loss = self._train(model, optimizer, train_pos_edge_index, data, x)
            auc, ap = self._test(
                data.test_pos_edge_index,
                data.test_neg_edge_index,
                model,
                x,
                train_pos_edge_index,
            )
            # print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}".format(epoch, auc, ap))

        print("loss:", loss)

        with torch.no_grad():
            self._embedding = {}
            encoded = model.encode(x, train_pos_edge_index).numpy()
            for i in range(num_nodes):
                self._embedding[i] = encoded[i, :]
