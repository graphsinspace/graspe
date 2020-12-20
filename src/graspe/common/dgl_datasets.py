from dgl.data import KarateClubDataset

from common.base.dataset import Dataset
from common.base.graph import Graph
from common.dgl_graph import DGLGraph


class KarateClub(Dataset):
    def load(self) -> Graph:
        self.ds = KarateClubDataset()
        g = DGLGraph.from_existing(self.ds[0])
        return g
