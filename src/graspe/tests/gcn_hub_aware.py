from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding
import sys

DATASETS = ['pubmed', 'citeseer', 'cora_ml', 'cora', 'dblp', 'karate_club_graph',
            'amazon_electronics_computers', 'amazon_electronics_photo']


def compare_gcns():
    datasets = DatasetPool.get_datasets()
    print(datasets)
    for dataset_name in DATASETS:
        g = DatasetPool.load(dataset_name)
        for epochs in [100,200]:
            for bad_aware in [True, False]:
                e = GCNEmbedding(
                    g,
                    d=100,
                    epochs=epochs,
                    lr=0.05,
                    layer_configuration=(128, 256, 128),
                    act_fn="tanh",
                    badness_aware=bad_aware
                )
                file_name = '{}_gcn_embedding_epochs={}_badness_aware={}'.format(dataset_name, int(epochs), bad_aware)
                sys.stdout = open(file_name, 'w')
                e.embed()


if __name__ == '__main__':
    compare_gcns()