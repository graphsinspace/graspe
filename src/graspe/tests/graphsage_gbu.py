import sys
from common.dataset_pool import DatasetPool
from embeddings.embedding_graphsage import GraphSageEmbedding


DATASETS = ['pubmed', 'citeseer', 'cora_ml', 'cora', 'dblp', 'karate_club_graph',
            'amazon_electronics_computers', 'amazon_electronics_photo']
FILE_PATH = '/home/stamenkovicd/graphsage_hub_aware_res/'


def compare_graphsages():
    for dataset_name in DATASETS:
        g = DatasetPool.load(dataset_name)
        for epochs in [100, 200]:
                e = GraphSageEmbedding(
                    g,
                    d=100,
                    epochs=epochs,
                    lr=0.05,
                    layer_configuration=(128, 256, 128),
                    act_fn="tanh",
                    hub_aware=True
                )
                file_name = '{}_graphsage_embedding_epochs={}_hubness_aware=true'.format(dataset_name, int(epochs))
                print('Results saved at:', FILE_PATH + file_name)
                sys.stdout = open(FILE_PATH + file_name, 'w')
                e.embed()


if __name__ == '__main__':
    compare_graphsages()
