import os
import networkx as nx
from common.graph_loaders import load_from_file
from common.graph import Graph

class DatasetPool:
    __pool = None

    @staticmethod
    def load(name):
        DatasetPool.__init_pool()
        if name in DatasetPool.__pool:
            method, parameter = DatasetPool.__pool[name]
            return method(parameter)
        return None

    def get_datasets():
        DatasetPool.__init_pool()
        return DatasetPool.__pool.keys()

    @staticmethod
    def __init_pool():
        if DatasetPool.__pool != None:
            return
        DatasetPool.__pool = {}        
        
        # Init from "data" directory.
        file_dataset_labels = {
            'cora_ml': 'labels'
        }
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'data')
        for f in os.listdir(base_path):
            path = os.path.join(base_path, f)
            if os.path.isfile(path) and f[0] != '.':
                name, ext = os.path.splitext(f)
                DatasetPool.__pool[name] = (lambda x : load_from_file(x, file_dataset_labels.get(os.path.splitext(os.path.basename(x))[0])), path)

        # Init form networkx
        nx_dataset_labels = {
            'karate_club_graph': 'club',
            'davis_southern_women_graph': None,
            'florentine_families_graph': None,
            'les_miserables_graph': None
        }
        for dataset in nx_dataset_labels:
            DatasetPool.__pool[dataset] = (lambda x : Graph(getattr(nx, x)(), nx_dataset_labels[x]), dataset)
