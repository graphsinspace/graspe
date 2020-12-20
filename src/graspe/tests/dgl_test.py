import pytest
import torch

from common.base.graph import Graph
from common.dgl_datasets import KarateClub
from common.dgl_graph import DGLGraph
from common.file_datasets import CSVDataset
from embeddings.gcn_embedding import GCNEmbedding


class TestDGL:
    def test_load_dgl_karate_club(self):
        kc: Graph = KarateClub().load()
        print(kc.nodes())
        print(kc.edges())
        assert kc

    def test_create_graph(self):
        g = DGLGraph(torch.tensor([2, 5, 3]), torch.tensor([3, 5, 0]))
        assert g

    def test_csv_load(self):
        csvd = CSVDataset(edges_path="tests/data/example.csv")
        g = csvd.load()
        print(g.nodes(), g.edges())
        assert g

    def test_to_networkx(self):
        g = DGLGraph(torch.tensor([2, 5, 3]), torch.tensor([3, 5, 0]))
        assert g.to_networkx().__class__.__name__ == "MultiDiGraph"

    def test_embed(self):
        g = KarateClub().load()
        dimension = 5
        embedding = GCNEmbedding(g, dimension)
        embedding.embed(args={
            "labeled_nodes": torch.tensor([0, 33]),
            "labels": torch.tensor([0, 1]),
            "epochs": 50
        })
        print(embedding[0])
        print(type(embedding[0]))
        assert isinstance(embedding[0], torch.Tensor) and len(embedding[0]) == dimension
