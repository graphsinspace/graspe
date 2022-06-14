import sys
import itertools
import pickle
import os
import pandas as pd
from collections import namedtuple
from common.dataset_pool import DatasetPool
from embeddings.embedding_graphsage import GraphSageEmbedding, GraphSageEmbeddingHubAware, \
                                           GraphSageEmbeddingBadAware, GraphSageEmbeddingNCLID
import sys


Model = namedtuple('Model', ['name', 'algo'])
MODELS = [
    Model('gs', GraphSageEmbedding),
    Model('gs_hub_aware', GraphSageEmbeddingHubAware),
    Model('gs_bad_aware', GraphSageEmbeddingBadAware),
    Model('gs_nclid', GraphSageEmbeddingNCLID)
] 
DATASETS = ['karate_club_graph', 'pubmed', 'citeseer', 'cora_ml', 'cora', 'dblp']
# DATASETS = ['amazon_electronics_computers', 'amazon_electronics_photo']


def produce_configs():
    configs = list(
        itertools.product(
            ["tanh", "relu"],                                   # 0 act_fn
            [0.01, 0.1],                                        # 1 learning rate
            [100, 200],                                         # 2 epochs
            [(128,), (128, 128), (256, 256), (256, 512, 256)],  # 3 layer configs
            [100],                                              # 4 dim
        )
    )
    return configs

if __name__ == '__main__':
    configs = produce_configs()
    
    for dataset in DATASETS:

        cols = ['dataset','algo', 'dim', 'layer_config', 'epochs', 'act_fn', 'lr', 'acc', 'prec', 'rec', 'f1']
        final_res = pd.DataFrame(columns=cols)
        g = DatasetPool.load(dataset)
        
        for model in MODELS:
            for config in configs:
                print('config\n', config)
                print('model.algo', model.algo)
                print('layer_config', config[3])
                sys.stdout.flush()
                if model.algo != 'gs_nclid':
                    e = model.algo(
                        g,
                        d=config[4],
                        epochs=config[2],
                        lr=config[1],
                        layer_configuration=config[3],
                        act_fn=config[0]
                    )
                else:
                    e = model.algo(
                        g,
                        d=config[4],
                        epochs=config[2],
                        lr=config[1],
                        layer_configuration=config[3],
                        act_fn=config[0],
                        dataset_name=dataset
                    )
                acc, prec, rec, f1 = e.embed()
                current_res = pd.DataFrame(
                    [[dataset, model.name, config[4], config[3], config[2], config[0], config[1],  acc, prec, rec, f1]],
                    columns=cols
                )
                final_res = pd.concat([final_res, current_res], axis=0)
        with open('/home/stamenkovicd/graphsage_tuning_res_{}.pkl'.format(dataset), 'wb') as f:
            pickle.dump(final_res, f)