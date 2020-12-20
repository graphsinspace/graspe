import torch
from dgl.data import KarateClubDataset

from common.base.dataset import Dataset
from common.base.graph import Graph
from common.dgl_graph import DGLGraph
import csv


# TODO: add loading of features


class CSVDataset(Dataset):
    def __init__(self, edges_path, type="dgl"):
        super().__init__()
        self.edges_path = edges_path
        self.graph_type = type

    def load(self) -> Graph:
        if self.graph_type == "dgl":
            with open(self.edges_path, "r") as edges:
                rows = list(csv.reader(edges))
                left_side = [int(i) for i in rows[0]]
                right_side = [int(i) for i in rows[1]]
                g = DGLGraph(torch.tensor(left_side), torch.tensor(right_side))
                return g
