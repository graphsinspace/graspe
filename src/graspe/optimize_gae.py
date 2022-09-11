import sys
import itertools
import pickle
import statistics
import pandas as pd
from collections import namedtuple
from common.dataset_pool import DatasetPool
from embeddings.embedding_gae import GAEEmbedding, GAEEmbeddingBadAware, GAEEmbeddingNCLIDAware, GAEEmbeddingHubAware

Model = namedtuple('Model', ['name', 'algo'])
MODELS = [
    Model('GAE', GAEEmbedding),
    Model('GAE Hub Aware', GAEEmbeddingHubAware),
    Model('GAE Bad Aware', GAEEmbeddingBadAware),
    Model('GAE NCLID', GAEEmbeddingNCLIDAware)

]

FILE_PATH = '/home/stamenkovicd/gs_comparison/'
DATASETS = ['citeseer', 'cora_ml', 'cora', 'dblp', 'karate_club_graph']
# DATASETS = ['amazon_electronics_computers', 'amazon_electronics_photo']


def produce_configs():
    configs = list(
        itertools.product(
            ["tanh", "relu"],  # act_fn
            [0.01, 0.1],  # learning rate
            [100, 200],  # epochs
            [(128,), (128, 128), (256, 256), (256, 512, 256)],  # layer configs
            [100],  # dim
        )
    )
    return configs


if __name__ == '__main__':
    configs = produce_configs()

    for dataset in DATASETS:

        cols = ['dataset', 'algo', 'dim', 'layer_config', 'epochs', 'act_fn', 'lr', 'map', 'recall', 'f1']
        final_res = pd.DataFrame(columns=cols)
        g = DatasetPool.load(dataset)
        g_undirected = g.to_undirected()
        g_undirected.remove_selfloop_edges()

        for model in MODELS:
            for config in configs:
                if model.algo != 'GAE NCLID':
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
                e.embed()
                recg = e.reconstruct(g_undirected.edges_cnt())
                avg_map, maps = g_undirected.map_value(recg)
                avg_recall, recalls = g_undirected.recall(recg)
                f1 = [
                    (
                        (2 * maps[node] * recalls[node]) / (maps[node] + recalls[node])
                        if maps[node] + recalls[node] != 0
                        else 0
                    )
                    for node in maps
                ]
                avg_f1 = statistics.mean(f1)

                current_res = pd.DataFrame(
                    [[
                        dataset,        # dataset
                        model.name,     # algo
                        config[4],      # dim
                        config[3],      # layer_config
                        config[2],      # epochs
                        config[0],      # act_fn
                        config[1],      # lr
                        avg_map,        # map
                        avg_recall,     # recall
                        avg_f1          # f1
                    ]],
                    columns=cols
                )
                final_res = pd.concat([final_res, current_res], axis=0)
        with open('/home/stamendu/gae_tuning_res/gae_tuning_res_{}.pkl'.format(dataset), 'wb') as f:
            pickle.dump(final_res, f)
