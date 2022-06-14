import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch.nn import Sequential
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.utils import train_test_split_edges
from embeddings.base.embedding import Embedding
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn.inits import reset



class GAEBase(torch.nn.Module):
    def __init__(self, encoder, decoder=None):
        super(GAEBase, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAEBase.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
    
    @abstractmethod
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pass
    
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    
class GAE(GAEBase):
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)
    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        return pos_loss + neg_loss
    

class GAEHubAware(GAEBase):
    def __init__(self, hubs_combine, hub_vector, encoder, decoder=None):
        super().__init__(encoder, decoder)
        assert hubs_combine in {'add', 'multiply'}, f'expected "add" or "multiply", got: {hubs_combine}'
        self.hubs_combine = hubs_combine
        self.hub_vector = hub_vector
    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        hub_vector_pos = hub_vector[pos_edge_index]
        hub_vector_neg = hub_vector[neg_edge_index]
        if self.hubs_combine == 'add':
            hub_vector_pos = hub_vector_pos.sum(axis=0)
            hub_vector_neg = hub_vector_neg.sum(axis=0)
        else:
            hub_vector_pos = hub_vector_pos[0] * hub_vector_pos[1]
            hub_vector_neg = hub_vector_neg[0] * hub_vector_neg[1]
        
        hub_vector_pos /= hub_vector_pos.norm()
        hub_vector_neg /= hub_vector_neg.norm()
        
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15)
        pos_loss = torch.mult(pos_loss, hub_vector_pos).mean()
        
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15)
        neg_loss = torch.mult(neg_loss, hub_vector_neg).mean()

        return pos_loss + neg_loss

    
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
        self.__logstd__ = self.__logstd__.clamp(max=10)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
