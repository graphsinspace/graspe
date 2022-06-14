import sys
import pickle
import torch
from common.dataset_pool import DatasetPool
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch, NCLIDEstimator

DATASETS = ['karate_club_graph', 'pubmed', 'citeseer', 'cora_ml', 'cora', 'dblp',
            'amazon_electronics_computers', 'amazon_electronics_photo']
FILE_PATH = '/home/stamenkovicd/nclids/'

def produce_nclids():
    for dataset_name in DATASETS:
        g = DatasetPool.load(dataset_name)
        print('... Calculating NCLIDs for {} dataset ...'.format(dataset_name))
        nclid = NCLIDEstimator(g, alpha=1)
        nclid.estimate_lids()
        nclids_tensor = torch.Tensor([nclid.get_lid(node[0]) for node in g.nodes()])
        file_name = '{}_nclids_tensor.pkl'.format(dataset_name)
        with open(FILE_PATH + file_name, 'wb') as file:
            pickle.dump(nclids_tensor, file)
        
if __name__ == '__main__':
    produce_nclids()
