import sys
from collections import namedtuple
from common.dataset_pool import DatasetPool
from embeddings.embedding_graphsage import GraphSageEmbedding, GraphSageEmbeddingHubAware, \
                                           GraphSageEmbeddingBadAware, GraphSageEmbeddingNCLID

DATASETS = ['pubmed', 'citeseer', 'cora_ml', 'cora', 'dblp', 'karate_club_graph',
            'amazon_electronics_computers', 'amazon_electronics_photo']
FILE_PATH = '/home/stamenkovicd/gs_comparison/'
Model = namedtuple('Model', ['name', 'algo'])
MODELS = [
    Model('GraphSAGE', GraphSageEmbedding),
    Model('GraphSAGE Hub Aware', GraphSageEmbeddingHubAware),
    Model('GraphSAGE Bad Aware', GraphSageEmbeddingBadAware),
    Model('GraphSAGE NCLID', GraphSageEmbeddingNCLID)
] 

def compare_graphsages():
    for dataset_name in DATASETS:
        g = DatasetPool.load(dataset_name)
        for epochs in [100, 200]:
            file_name = 'graphsage_comparison_{}_epochs_{}.txt'.format(dataset_name, epochs)
            print('Results saved at:', FILE_PATH + file_name)
            sys.stdout = open(FILE_PATH + file_name, 'w')
            for model in MODELS:
                print('\n' + model.name + '\n')
                e = model.algo(
                    g,
                    d=100,
                    epochs=epochs,
                    lr=0.05,
                    layer_configuration=(128, 256, 128),
                    act_fn="tanh"
                )
                e.embed()


if __name__ == '__main__':
    compare_graphsages()
