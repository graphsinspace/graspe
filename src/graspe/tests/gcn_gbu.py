import sys
from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


DATASETS = ['pubmed', 'citeseer', 'cora_ml', 'cora', 'dblp', 'karate_club_graph',
            'amazon_electronics_computers', 'amazon_electronics_photo']
FILE_PATH = '/home/stamenkovicd/gcn_gbu/'


def compare_gcns():
    for dataset_name in DATASETS:
        g = DatasetPool.load(dataset_name)
        for epochs in [100, 200]:
            for badness_aware in [True, False]:
                e = GCNEmbedding(
                    g,
                    d=100,
                    epochs=epochs,
                    lr=0.05,
                    layer_configuration=(128, 256, 128),
                    act_fn="tanh",
                    badness_aware=badness_aware
                )
                file_name = '{}_gcn_embedding_epochs={}_badness_aware={}'.format(
                    dataset_name, int(epochs), badness_aware)
                print('Results saved at:', FILE_PATH + file_name)
                sys.stdout = open(FILE_PATH + file_name, 'w')
                e.embed()


if __name__ == '__main__':
    compare_gcns()
