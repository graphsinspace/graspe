import argparse
import torch

from common.dataset_pool import DatasetPool
from common.graph_loaders import load_from_file


def load_graph(g):
    graph = DatasetPool.load(g)
    if graph == None:
        graph = load_from_file(g)
    if graph == None:
        raise Exception("invalid graph")
    return graph


def list_datasets(args):
    print(", ".join(DatasetPool.get_datasets()))


def embed(args):
    g = load_graph(args.graph)
    d = int(args.dimensions)
    o = args.out
    embedding = None
    if args.algorithm == "gcn":
        e = int(args.epochs)
        from embeddings.embedding_gcn import GCNEmbedding

        embedding = GCNEmbedding(g, d, e)
    elif args.algorithm == 'sdne':
        layers = args.layers
        alpha = args.alpha
        beta = args.beta
        nu1 = args.nu1
        nu2 = args.nu2
        bs = args.bs
        epochs = args.epochs
        verbose = args.verbose
        if len(layers) == 0:
            layers.append(4*d)
            layers.append(2*d)
        from embeddings.embedding_sdne import SDNEEmbedding
        embedding = SDNEEmbedding(g, d, layers, alpha, beta, nu1, nu2, bs, epochs, verbose)
        
    if args.algorithm == 'deep_walk':
        e = int(args.epochs)
        wn = int(args.walk_number)
        wl = int(args.walk_length)
        w = int(args.workers)
        ws = int(args.window_size)
        lr = float(args.learning_rate)
        mc = int(args.min_count)
        s = int(args.seed)
        from embeddings.embedding_deepwalk import DeepWalkEmbedding
        embedding = DeepWalkEmbedding(g, d, wn, wl, w, ws, e, lr, mc, s)
        

    if args.algorithm == "gae":
        e = int(args.epochs)
        variational = bool(args.variational)
        linear = bool(args.linear)
        from embeddings.embedding_gae import GAEEmbedding

        embedding = GAEEmbedding(g, d, e, variational, linear)

        
    if embedding:
        embedding.embed()
        embedding.to_file(o)
        reconstruction = embedding.reconstruct(len(g.edges()))
        print("precission@k: " + str(g.link_precision(reconstruction)))
        print("map: " + str(g.map(reconstruction)))


if __name__ == "__main__":
    # Parsing arguments.
    parser = argparse.ArgumentParser(
        description="Graphs in Space: Graph Embeddings for Machine Learning on Complex Data -- Evaluation."
    )
    subparsers = parser.add_subparsers(
        title="actions", description="available actions", dest="action", required=True
    )

    # Action: list_datasets.
    parser_list_datasets = subparsers.add_parser(
        "list_datasets", help="list_datasets help"
    )

    # Action: embed.
    parser_embed = subparsers.add_parser("embed", help="list_datasets help")
    subparsers_embed = parser_embed.add_subparsers(
        title="algorithm",
        description="embedding algorithm",
        dest="algorithm",
        required=True,
    )

    # --- Embedding algorithms:

    # --- GCN
    parser_embed_gcn = subparsers_embed.add_parser("gcn", help="GCN embedding")
    # --- GCN arguments:
    # ------ Mutual arguments:
    parser_embed_gcn.add_argument(
        "-g",
        "--graph",
        help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
        "karate_club_graph).",
        required=True,
    )
    parser_embed_gcn.add_argument(
        "-d", "--dimensions", help="Dimensions of the embedding.", required=True
    )
    parser_embed_gcn.add_argument(
        "-o", "--out", help="Output file.", default="out.embedding"
    )
    # ------ GCN-specific arguments:
    parser_embed_gcn.add_argument(
        "-e", "--epochs", help="Number of epochs.", default=50
    )
    
    parser_embed_gcn.add_argument('-g', '--graph', help='Path to the graph, or name of the dataset from the dataset pool (e.g. '
                                              'karate_club_graph).', required=True)
    parser_embed_gcn.add_argument('-d', '--dimensions', help='Dimensions of the embedding.', required=True)
    parser_embed_gcn.add_argument('-o', '--out', help='Output file.', default='out.embedding')
    # ------ GCN-specific arguments:
    parser_embed_gcn.add_argument("-e", "--epochs", help="Number of epochs.", default=50)
    
    # --- SDNE
    parser_embed_sdne = subparsers_embed.add_parser('sdne', help='SDNE embedding')    
    # --- SDNE arguments:
    # ------ Mutual arguments:
    parser_embed_sdne.add_argument('-g', '--graph', help='Path to the graph, or name of the dataset from the dataset pool (e.g. '
                                              'karate_club_graph).', required=True)
    parser_embed_sdne.add_argument('-d', '--dimensions', help='Dimensions of the embedding.', required=True)
    parser_embed_sdne.add_argument('-o', '--out', help='Output file.', default='out.embedding')
    # ------ SDNE-specific arguments:
    parser_embed_sdne.add_argument("-l", "--layers", type=int, nargs='+', help="Layers structure.", default=[])
    parser_embed_sdne.add_argument("-a", "--alpha", type=float, help="Alpha parameter.", default=1e-5)
    parser_embed_sdne.add_argument("-b", "--beta", type=float, help="Beta parameter.", default=5.)
    parser_embed_sdne.add_argument("--nu1", type=float, help="nu1 parameter.", default=1e-6)
    parser_embed_sdne.add_argument("--nu2", type=float, help="nu2 parameter.", default=1e-6)
    parser_embed_sdne.add_argument("--bs", type=int, help="Batch size.", default=500)
    parser_embed_sdne.add_argument("-e", "--epochs", type=int, help="Number of epochs.", default=50)
    parser_embed_sdne.add_argument("-v", "--verbose", type=int, help="Verbose.", default=0)

    # ---
    parser_embed_gae = subparsers_embed.add_parser("gae", help="GAE embedding")
    # --- GAE arguments:
    # ------ Mutual arguments:
    parser_embed_gae.add_argument(
        "-g",
        "--graph",
        help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
        "karate_club_graph).",
        required=True,
    )
    parser_embed_gae.add_argument(
        "-d", "--dimensions", help="Dimensions of the embedding.", required=True
    )
    parser_embed_gae.add_argument(
        "-o", "--out", help="Output file.", default="out.embedding"
    )
    # ------ GAE-specific arguments:
    parser_embed_gae.add_argument(
        "-e", "--epochs", help="Number of epochs.", default=50
    )

    parser_embed_gae.add_argument(
        "-v", "--variational", action="store_true", help="Whether to use variational AEs", default=False
    )

    parser_embed_gae.add_argument(
        "-l", "--linear", action="store_true", help="Whether to use linear encoders", default=False
    )    

    # --- DeepWalk
    parser_embed_deep_walk = subparsers_embed.add_parser('deep_walk', help='DeepWalk embedding')

    parser_embed_deep_walk.add_argument('-g', '--graph', help='Path to the graph, or name of the dataset from the dataset pool (e.g. '
                                              'karate_club_graph).', required=True)
    parser_embed_deep_walk.add_argument('-d', '--dimensions', help='Dimensions of the embedding.', required=True)
    parser_embed_deep_walk.add_argument('-o', '--out', help='Output file.', default='out.embedding')
    parser_embed_deep_walk.add_argument('-e', '--epochs', help='Number of epochs.', default=50)
    parser_embed_deep_walk.add_argument('-wn', '--walk_number', help='Number of random walks.', default=10)
    parser_embed_deep_walk.add_argument('-wl', '--walk_length', help='Length of random walks.', default=80)
    parser_embed_deep_walk.add_argument('-w', '--workers', help='Number of cores.', default=4)
    parser_embed_deep_walk.add_argument('-ws', '--window_size', help='Matrix power order.', default=5)
    parser_embed_deep_walk.add_argument('-lr', '--learning_rate', help='HogWild! learning rate.', default=0.05)
    parser_embed_deep_walk.add_argument('-mc', '--min_count', help='Minimal count of node occurrences.', default=1)
    parser_embed_deep_walk.add_argument('-s', '--seed', help='Random seed value.', default=42)
    

    # Execute the action.
    args = parser.parse_args()
    globals().get(args.action)(args)
