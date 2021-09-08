import statistics
import json
from tqdm import tqdm
from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_graphsage import GraphSageEmbedding
from embeddings.embedding_gae import GAEEmbedding

DATASETS = [
        'karate_club_graph',
        'amazon_electronics_photo',
        'cora',
        'amazon_electronics_computers',
        'cora_ml',
        'citeseer',
        'dblp',
        'pubmed'
    ]


def compare_gcn_all(datasets):
    best_confs = [
         [200, 'relu', 0.1, 200, (256, 256)],
         [200, 'tanh', 0.1, 200, (256, 512, 256)],
         [200, 'tanh', 0.1, 200, (256, 512, 256)],
         [200, 'relu', 0.1, 200, (128,)],
         [200, 'tanh', 0.1, 100, (256, 512, 256)],
         [200, 'tanh', 0.1, 200, (256, 512, 256)],
         [200, 'relu', 0.1, 100, (128,)],
         [200, 'tanh', 0.1, 200, (256, 512, 256)]
    ]
    results = {}
    for i in tqdm(range(len(datasets))):
        dataset = datasets[i]
        results[dataset] = {}
        best_conf = best_confs[i]
        g = DatasetPool.load(dataset)
        g_undirected = g.to_undirected()
        g_undirected.remove_selfloop_edges()
        for hub_fn in ['identity', 'inverse', 'log', 'log_inverse']:
            e = GCNEmbedding(
                g,
                d=best_conf[0],
                epochs=best_conf[3],
                lr=best_conf[2],
                layer_configuration=best_conf[4],
                act_fn=best_conf[1],
                hub_aware=True,
                hub_fn=hub_fn
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
            results[dataset][hub_fn] = {
                'avg_map': avg_map,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1
            }
    return results


def compare_graphsage_all(datasets):
    best_confs = [
         [10, 'tanh', 0.1, 100, (128,)],
         [50, 'tanh', 0.01, 200, (256, 512, 256)],
         [25, 'tanh', 0.01, 200, (256, 512, 256)],
         [25, 'tanh', 0.01, 100, (256, 512, 256)],
         [100, 'tanh', 0.01, 100, (256, 256)],
         [10, 'relu', 0.1, 100, (128, 128)],
         [25, 'relu', 0.1, 100, (128, 128)],
         [10, 'tanh', 0.1, 100, (256, 512, 256)]
    ]
    results = {}
    for i in tqdm(range(len(datasets))):
        dataset = datasets[i]
        results[dataset] = {}
        best_conf = best_confs[i]
        g = DatasetPool.load(dataset)
        g_undirected = g.to_undirected()
        g_undirected.remove_selfloop_edges()
        for hub_fn in ['identity', 'inverse', 'log', 'log_inverse']:
            e = GraphSageEmbedding(
                g,
                d=best_conf[0],
                epochs=best_conf[3],
                lr=best_conf[2],
                layer_configuration=best_conf[4],
                act_fn=best_conf[1],
                hub_aware=True,
                hub_fn=hub_fn
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

            results[dataset][hub_fn] = {
                'avg_map': avg_map,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1
            }
    return results


def compare_gae_all(datasets):
    best_confs = [
         [200, 'relu', False, True, 0.1, 200, (256, 256)],
         [200, 'relu', False, True, 0.1, 200, (256, 512, 256)],
         [200, 'relu', False, True, 0.1, 200, (256, 512, 256)],
         [100, 'tanh', True, True, 0.1, 200, (256, 512, 256)],
         [100, 'relu', False, True, 0.1, 100, (256, 256)],
         [200, 'tanh', False, True, 0.1, 100, (256, 256)],
         [200, 'tanh', False, True, 0.01, 100, (256, 256)],
         [100, 'relu', True, True, 0.1, 200, (256, 512, 256)]
    ]
    results = {}
    for i in tqdm(range(len(datasets))):
        dataset = datasets[i]
        results[dataset] = {}
        best_conf = best_confs[i]
        g = DatasetPool.load(dataset)
        g_undirected = g.to_undirected()
        g_undirected.remove_selfloop_edges()
        for hub_fn in ['identity', 'inverse', 'log', 'log_inverse']:
            e = GAEEmbedding(
                g,
                d=best_conf[0],
                epochs=best_conf[5],
                variational=best_conf[2],
                linear=best_conf[3],
                lr=best_conf[4],
                layer_configuration=best_conf[6],
                act_fn=best_conf[1],
                hub_aware=True,
                hub_fn=hub_fn
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

            results[dataset][hub_fn] = {
                'avg_map': avg_map,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1
            }

    return results


if __name__ == '__main__':
    results_gcn = compare_gcn_all(DATASETS)
    with open('/home/dusan/graspe_gcn_res/gcn_hub_aware.json', 'w') as file:
        json.dump(results_gcn, file, indent=4)

    results_graphsage = compare_graphsage_all(DATASETS)
    with open('/home/dusan/graspe_graphsage_res/graphsage_hub_aware.json', 'w') as file:
        json.dump(results_graphsage, file, indent=4)

    results_gae = compare_gae_all(DATASETS)
    with open('/home/dusan/graspe_gae_res/gae_hub_aware.json', 'w') as file:
        json.dump(results_gae, file, indent=4)

