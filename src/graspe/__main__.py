import argparse
import os
import torch

from common.dataset_pool import DatasetPool
from common.graph_loaders import load_from_file
from embeddings.embfactory import LazyEmbFactory


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

    elif args.algorithm == "sdne":
        layers = args.layers
        alpha = args.alpha
        beta = args.beta
        nu1 = args.nu1
        nu2 = args.nu2
        bs = args.bs
        epochs = args.epochs
        verbose = args.verbose
        if len(layers) == 0:
            layers.append(4 * d)
            layers.append(2 * d)
        from embeddings.embedding_sdne import SDNEEmbedding

        embedding = SDNEEmbedding(
            g, d, layers, alpha, beta, nu1, nu2, bs, epochs, verbose
        )

    elif args.algorithm == "deep_walk":
        pn = int(args.path_number)
        pl = int(args.path_length)
        w = int(args.workers)
        ws = int(args.window_size)

        from embeddings.embedding_deepwalk import DeepWalkEmbedding

        embedding = DeepWalkEmbedding(g, d, pn, pl, w, ws)

    elif args.algorithm == "gae":
        e = int(args.epochs)
        variational = bool(args.variational)
        linear = bool(args.linear)
        from embeddings.embedding_gae import GAEEmbedding

        embedding = GAEEmbedding(g, d, e, variational, linear)

    elif args.algorithm == "node2vec":
        p = float(args.p)
        q = float(args.q)
        walk_length = int(args.walk_length)
        num_walks = int(args.num_walks)
        workers = int(args.workers)
        from embeddings.embedding_node2vec import Node2VecEmbedding

        embedding = Node2VecEmbedding(g, d, p, q, walk_length, num_walks, workers)

    if embedding:
        embedding.embed()
        embedding.to_file(o)
        reconstruction = embedding.reconstruct(len(g.edges()))
        print("precission@k: " + str(g.link_precision(reconstruction)))
        print("map: " + str(g.map(reconstruction)))


def batch_embed(args):
    g = load_graph(args.graph)
    for d in args.dimensions:
        emb_factory = LazyEmbFactory(g, d, preset=args.preset)
        for i in range(emb_factory.num_methods()):
            name = emb_factory.get_name(i)
            embedding = emb_factory.get_embedding(i)
            embedding.embed()
            embedding.to_file(
                os.path.join(
                    args.out, args.graph + "_d" + str(d) + "_" + name + ".embedding"
                )
            )


def classify(args):
    classes = None
    if args.classify == "kNN":
        from classifications.k_nearest_neighbors import KNN

        classes = KNN(args.embedding, int(args.k_neighbors))

    elif args.classify == "rf":
        from classifications.random_forest import RandomForest

        classes = RandomForest(args.embedding, int(args.n_estimators))

    elif args.classify == "svm":
        from classifications.support_vector_machines import SVM

        classes = SVM(args.embedding)

    elif args.classify == "nb":
        from classifications.naive_bayes import NaiveBayes

        classes = NaiveBayes(args.embedding)

    elif args.classify == "dt":
        from classifications.decision_tree import DecisionTree

        classes = DecisionTree(args.embedding)

    elif args.classify == "nn":
        from classifications.neural_network import NeuralNetworkClassification

        classes = NeuralNetworkClassification(args.embedding, int(args.epochs))

    if classes:
        classes.classify()


def hub_eval(args):
    import evaluation.hub_focused_eval as he

    out = args.out
    if not out:
        out = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "reports",
            "figures",
            "hub_focused_eval",
        )
    he.eval(args.dimensions, out, args.graph)


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
    parser_embed = subparsers.add_parser("embed", help="Embed a graph")
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

    # --- SDNE
    parser_embed_sdne = subparsers_embed.add_parser("sdne", help="SDNE embedding")
    # --- SDNE arguments:
    # ------ Mutual arguments:
    parser_embed_sdne.add_argument(
        "-g",
        "--graph",
        help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
        "karate_club_graph).",
        required=True,
    )
    parser_embed_sdne.add_argument(
        "-d", "--dimensions", help="Dimensions of the embedding.", required=True
    )
    parser_embed_sdne.add_argument(
        "-o", "--out", help="Output file.", default="out.embedding"
    )
    # ------ SDNE-specific arguments:
    parser_embed_sdne.add_argument(
        "-l", "--layers", type=int, nargs="+", help="Layers structure.", default=[]
    )
    parser_embed_sdne.add_argument(
        "-a", "--alpha", type=float, help="Alpha parameter.", default=1
    )
    parser_embed_sdne.add_argument(
        "-b", "--beta", type=float, help="Beta parameter.", default=5.0
    )
    parser_embed_sdne.add_argument(
        "--nu1", type=float, help="nu1 parameter.", default=1e-6
    )
    parser_embed_sdne.add_argument(
        "--nu2", type=float, help="nu2 parameter.", default=1e-6
    )
    parser_embed_sdne.add_argument(
        "--bs", type=int, help="Batch size (not yet implemented).", default=1024
    )
    parser_embed_sdne.add_argument(
        "-e", "--epochs", type=int, help="Number of epochs.", default=50
    )
    parser_embed_sdne.add_argument(
        "-v", "--verbose", type=int, help="Verbose.", default=0
    )

    # --- GAE
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
        "-v",
        "--variational",
        action="store_true",
        help="Whether to use variational AEs",
        default=False,
    )

    parser_embed_gae.add_argument(
        "-l",
        "--linear",
        action="store_true",
        help="Whether to use linear encoders",
        default=False,
    )

    # --- DeepWalk
    parser_embed_deep_walk = subparsers_embed.add_parser(
        "deep_walk", help="DeepWalk embedding"
    )

    parser_embed_deep_walk.add_argument(
        "-g",
        "--graph",
        help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
        "karate_club_graph).",
        required=True,
    )
    parser_embed_deep_walk.add_argument(
        "-d", "--dimensions", help="Dimensions of the embedding.", required=True
    )
    parser_embed_deep_walk.add_argument(
        "-o", "--out", help="Output file.", default="out.embedding"
    )
    parser_embed_deep_walk.add_argument(
        "--path_number", help="Number of random path.", default=10
    )
    parser_embed_deep_walk.add_argument(
        "--path_length", help="Length of random path.", default=80
    )
    parser_embed_deep_walk.add_argument("--workers", help="Number of cores.", default=4)
    parser_embed_deep_walk.add_argument(
        "--window_size", help="Matrix power order.", default=5
    )

    # -------- Node2Vec
    parser_embed_node2vec = subparsers_embed.add_parser(
        "node2vec", help="Node2Vec embedding"
    )
    parser_embed_node2vec.add_argument(
        "-g",
        "--graph",
        help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
        "karate_club_graph).",
        required=True,
    )
    parser_embed_node2vec.add_argument(
        "-p", "--p", help="Return hyper parameter.", required=True
    )
    parser_embed_node2vec.add_argument(
        "-q", "--q", help="Inout hyper parameter.", required=True
    )
    parser_embed_node2vec.add_argument(
        "-d", "--dimensions", help="Dimensions of the embedding.", required=True
    )
    parser_embed_node2vec.add_argument(
        "-o", "--out", help="Output file.", default="out.embedding"
    )
    parser_embed_node2vec.add_argument(
        "-walk_length", "--walk_length", help="Length of random walks.", default=10
    )
    parser_embed_node2vec.add_argument(
        "-num_walks", "--num_walks", help="Number of random walks.", default=200
    )
    parser_embed_node2vec.add_argument(
        "-workers",
        "--workers",
        help="Number of workers for parallel execution.",
        default=1,
    )

    # Action: batch_embed.

    parser_batch_embed = subparsers.add_parser(
        "batch_embed", help="Create a number of embeddings for a graph."
    )
    parser_batch_embed.add_argument(
        "-g",
        "--graph",
        help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
        "karate_club_graph).",
        required=True,
    )
    parser_batch_embed.add_argument(
        "-d",
        "--dimensions",
        help="Dimensions of the embedding.",
        type=int,
        nargs="+",
        required=True,
    )
    parser_batch_embed.add_argument(
        "-p", "--preset", help="Algorithms preset name.", default="_"
    )
    parser_batch_embed.add_argument(
        "-o", "--out", help="Output directory.", required=True
    )

    # Action: classify.
    parser_classify = subparsers.add_parser("classify", help="Do classification")
    subparsers_classify = parser_classify.add_subparsers(
        title="classify",
        dest="classify",
        required=True,
    )

    # kNN
    parser_classify_knn = subparsers_classify.add_parser(
        "kNN", help="k Nearest Neighbors classification."
    )
    parser_classify_knn.add_argument(
        "-e", "--embedding", help="Path to the embedding file.", required=True
    )
    parser_classify_knn.add_argument(
        "-k",
        "--k_neighbors",
        help="Number of neighbors to use by default for kneighbors queries.",
        required=True,
    )

    # RandomForest
    parser_classify_rf = subparsers_classify.add_parser(
        "rf", help="Random Forest classification."
    )
    parser_classify_rf.add_argument(
        "-e", "--embedding", help="Path to the embedding file.", required=True
    )
    parser_classify_rf.add_argument(
        "-n", "--n_estimators", help="The number of trees in the forest.", required=True
    )

    # SVM
    parser_classify_svm = subparsers_classify.add_parser(
        "svm", help="Support Vector Machines classification."
    )
    parser_classify_svm.add_argument(
        "-e", "--embedding", help="Path to the embedding file.", required=True
    )

    # NaiveBayes
    parser_classify_nb = subparsers_classify.add_parser(
        "nb", help="Naive Bayes classification (Gaussian)."
    )
    parser_classify_nb.add_argument(
        "-e", "--embedding", help="Path to the embedding file.", required=True
    )

    # DecisionTree
    parser_classify_dt = subparsers_classify.add_parser(
        "dt", help="Decision Tree classification."
    )
    parser_classify_dt.add_argument(
        "-e", "--embedding", help="Path to the embedding file.", required=True
    )

    # NeuralNetwork
    parser_classify_nn = subparsers_classify.add_parser(
        "nn", help="Neural Network classification."
    )
    parser_classify_nn.add_argument(
        "-em", "--embedding", help="Path to the embedding file.", required=True
    )
    parser_classify_nn.add_argument(
        "-ep", "--epochs", help="Number of epochs.", required=True
    )

    # Action: hub_eval.
    parser_hub_eval = subparsers.add_parser("hub_eval", help="hub_eval help")
    parser_hub_eval.add_argument(
        "-g",
        "--graph",
        help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
        "karate_club_graph).",
        default=None,
    )
    parser_hub_eval.add_argument(
        "-d",
        "--dimensions",
        help="Dimensions of the embedding.",
        required=True,
        type=int,
    )
    parser_hub_eval.add_argument("-o", "--out", help="Directory for the figures.")

    # Execute the action.
    args = parser.parse_args()
    globals().get(args.action)(args)
