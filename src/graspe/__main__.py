import argparse
import torch

from common.dataset_pool import DatasetPool
from common.graph_loaders import load_from_file
from embeddings.embedding_gcn import GCNEmbedding

def load_graph(g):
    graph = DatasetPool.load(g)
    if graph == None:
        graph = load_from_file(g)
    if graph == None:
        raise Exception('invalid graph')
    return graph

def list_datasets(args):
    print(", ".join(DatasetPool.get_datasets()))

def embed(args):
    g = load_graph(args.graph)
    d = int(args.dimensions)
    o = args.out
    embedding = None
    if args.algorithm == 'gcn':
        e = int(args.epochs)
        embedding = GCNEmbedding(g, d, e)
    if embedding:
        embedding.embed()
        embedding.to_file(o)

if __name__ == "__main__":
    # Parsing arguments.
    parser = argparse.ArgumentParser(
        description="Graphs in Space: Graph Embeddings for Machine Learning on Complex Data -- Evaluation."
    )
    subparsers = parser.add_subparsers(title='actions',
                                       description='available actions',
                                       dest='action',
                                       required=True)

    # Action: list_datasets.
    parser_list_datasets = subparsers.add_parser('list_datasets', help='list_datasets help')

    # Action: embed.
    parser_embed = subparsers.add_parser('embed', help='list_datasets help')
    subparsers_embed = parser_embed.add_subparsers(title='algorithm',
                                       description='embedding algorithm',
                                       dest='algorithm',
                                       required=True)
    
    # --- Embedding algorithms:
    
    # --- GCN
    parser_embed_gcn = subparsers_embed.add_parser('gcn', help='GCN embedding')    
    # --- GCN arguments:
    # ------ Mutual arguments:
    parser_embed_gcn.add_argument('-g', '--graph', help='Path to the graph, or name of the dataset from the dataset pool (e.g. '
                                              'karate_club_graph).', required=True)
    parser_embed_gcn.add_argument('-d', '--dimensions', help='Dimensions of the embedding.', required=True)
    parser_embed_gcn.add_argument('-o', '--out', help='Output file.', default='out.embedding')
    # ------ GCN-specific arguments:
    parser_embed_gcn.add_argument("-e", "--epochs", help="Number of epochs.", default=50)
    
    # Execute the action.
    args = parser.parse_args()
    globals().get(args.action)(args)