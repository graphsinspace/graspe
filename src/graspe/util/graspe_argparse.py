import argparse


def build_argparser():
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
    parser_list_datasets.add_argument(
        "-d",
        "--detailed",
        help="Detailed overview of available datasets.",
        action="store_true",
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

    argparse_gcn(subparsers_embed)

    argparse_sdne(subparsers_embed)

    argparse_gae(subparsers_embed)

    argparse_deepwalk(subparsers_embed)

    argparse_n2v(subparsers_embed)

    # Action: batch_embed.
    parser_batch_embed = subparsers.add_parser(
        "batch_embed", help="Create a number of embeddings for a graph."
    )
    parser_batch_embed.add_argument(
        "-g",
        "--graphs",
        help="Path to the graphs, or names of the datasets from the dataset pool (e.g. "
        "karate_club_graph).",
        nargs="+",
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
        "-a",
        "--algs",
        help="Algorithms",
        nargs="+",
        default=None,
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

    # Classifiers:
    argparse_classifiers(subparsers_classify)

    # Action: hub_eval.
    parser_hub_eval = subparsers.add_parser("hub_eval", help="hub_eval help")
    parser_hub_eval.add_argument(
        "-g",
        "--graph",
        help="Name of the dataset from the dataset pool",
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
    parser_hub_eval.add_argument(
        "-i", "--input", help="Directory where the embeddings are stored."
    )
    parser_hub_eval.add_argument(
        "-k",
        help="K value for kNN graph.",
        required=True,
        type=int,
    )
    parser_hub_eval.add_argument(
        "-p", "--preset", help="Algorithms preset name.", default="_"
    )
    parser_hub_eval.add_argument(
        "-a",
        "--algs",
        help="Algorithms",
        nargs="+",
        default=None,
    )

    # Action: generate_graphs
    parser_generate_graphs = subparsers.add_parser(
        "generate_graphs", help="generate_graphs help"
    )
    parser_generate_graphs.add_argument(
        "-n",
        "--nvals",
        help="Number of nodes",
        type=int,
        nargs="+",
        default=[100],
    )
    parser_generate_graphs.add_argument(
        "--kmvals",
        help="Values k (for newman-watts-strogatz) and values m (for barabasi-albert and powerlaw-cluster).",
        type=int,
        nargs="+",
        default=[5],
    )
    parser_generate_graphs.add_argument("-o", "--out", help="Output directory.")

    # Action: experimenter
    parser_experimenter = subparsers.add_parser(
        "experimenter", help="experimenter help"
    )
    parser_experimenter.add_argument(
        "-g",
        "--graphs",
        help="Name of the datasets from the dataset pool",
        nargs="+",
    )
    parser_experimenter.add_argument(
        "-d",
        "--dimension",
        help="Dimension of the embedding.",
        required=True,
        type=int,
    )
    parser_experimenter.add_argument(
        "-o",
        "--out",
        help="Output path. If not given, the results will be printed only in the standard output.",
        default=None,
    )
    parser_experimenter.add_argument(
        "-c", "--cache", help="Directory where the embeddings are stored.", default=None
    )
    parser_experimenter.add_argument(
        "-p", "--preset", help="Algorithms preset name.", default="_"
    )
    parser_experimenter.add_argument(
        "-a",
        "--algs",
        help="Algorithms",
        nargs="+",
        default=None,
    )

    return parser


def argparse_classifiers(subparsers_classify):
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


def argparse_n2v(subparsers_embed):
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
        "-walk_length", "--walk_length", help="Length of random walks.", default=80
    )
    parser_embed_node2vec.add_argument(
        "-num_walks", "--num_walks", help="Number of random walks.", default=10
    )
    parser_embed_node2vec.add_argument(
        "-workers",
        "--workers",
        help="Number of workers for parallel execution.",
        default=10,
    )


def argparse_deepwalk(subparsers_embed):
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


def argparse_gae(subparsers_embed):
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


def argparse_sdne(subparsers_embed):
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


def argparse_gcn(subparsers_embed):
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
