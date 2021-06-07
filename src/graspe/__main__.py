import os
import sys

from common.dataset_pool import DatasetPool
from common.graph_loaders import load_from_file
from embeddings.emb_factory import LazyEmbFactory
from util.graspe_argparse import build_argparser
from tqdm import tqdm


def load_graph(g):
    graph = DatasetPool.load(g)
    if not graph:
        graph = load_from_file(g, "")
    if not graph:
        raise Exception("invalid graph")
    return graph


def list_datasets(args):
    if args.detailed:
        for dataset in DatasetPool.get_datasets():
            g = DatasetPool.load(dataset)
            nodes_cnt = g.nodes_cnt()
            edges_cnt = g.edges_cnt()
            density = edges_cnt / ((nodes_cnt * (nodes_cnt - 1)) / 2)
            avg_h = edges_cnt / nodes_cnt
            h = g.get_hubness()
            max_h = h if h is int else max(h.values())
            print("Name: {}".format(dataset))
            print("Nodes count: {}".format(nodes_cnt))
            print("Edges count: {}".format(edges_cnt))
            print("Edges density: {}".format(density))
            print("Avg hubness: {}".format(avg_h))
            print("Max hubness: {}".format(max_h))
            print("Max hubness (normalized): {}".format(max_h / avg_h))
            print("=============================================")
    else:
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
        g_undirected = g.to_undirected()
        reconstruction = embedding.reconstruct(len(g_undirected.edges()))
        avg_map, maps = g_undirected.map_value(reconstruction)
        avg_recall, recalls = g_undirected.recall(reconstruction)
        print("precission@k: " + str(g_undirected.link_precision(reconstruction)))
        print("map: " + str(avg_map))
        print("recall: " + str(avg_recall))


def batch_embed(args):
    if args.graphs == ["all"]:
        args.graphs = DatasetPool.get_datasets()

    for g_name in tqdm(args.graphs):
        try:
            print("Batch embedding for ", g_name)
            g = load_graph(g_name)
            for d in tqdm(args.dimensions):
                emb_factory = LazyEmbFactory(g, d, preset=args.preset, algs=args.algs)
                for i in tqdm(range(emb_factory.num_methods())):
                    embedding = emb_factory.get_embedding(i)
                    embedding.embed()
                    embedding.to_file(
                        os.path.join(
                            args.out, emb_factory.get_full_name(g_name, i) + ".embedding"
                        )
                    )
        except Exception as e:
            print(f"Batch embedding for {g_name}, {args.preset} failed!")
            print(str(e))
            print(sys.gettrace())


def classify(args):
    classes = None
    if args.classify == "kNN":
        from evaluation.classifications.skl_classifiers import KNN

        classes = KNN(args.embedding, int(args.k_neighbors))

    elif args.classify == "rf":
        from evaluation.classifications.skl_classifiers import RandomForest

        classes = RandomForest(args.embedding, int(args.n_estimators))

    elif args.classify == "svm":
        from evaluation.classifications.skl_classifiers import SVM

        classes = SVM(args.embedding)

    elif args.classify == "nb":
        from evaluation.classifications.skl_classifiers import NaiveBayes

        classes = NaiveBayes(args.embedding)

    elif args.classify == "dt":
        from evaluation.classifications.skl_classifiers import DecisionTree

        classes = DecisionTree(args.embedding)

    elif args.classify == "nn":
        from evaluation.classifications.neural_network import (
            NeuralNetworkClassification,
        )

        classes = NeuralNetworkClassification(args.embedding, int(args.epochs))

    if classes:
        classes.classify()


def hub_eval(args):
    import evaluation.hub_focused_eval as he

    output_path = args.out
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "reports",
            "hub_focused_eval",
        )
    input_path = args.input
    if not input_path:
        input_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "embeddings"
        )
    he.hubness_stats(
        input_path,
        args.dimensions,
        output_path,
        args.graph,
        args.preset,
        args.algs,
        args.k,
    )


def generate_graphs(args):
    out = (
        args.out
        if args.out
        else os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )
    DatasetPool.generate_random_graphs(args.nvals, args.kmvals, out)


def experimenter(args):
    from evaluation.experimenter import Experimenter

    experimenter = Experimenter(
        args.graphs, args.dimension, args.preset, args.algs, args.cache
    )
    print(experimenter.run(args.out))


if __name__ == "__main__":
    parser = build_argparser()

    # Execute the action.
    args = parser.parse_args()
    globals().get(args.action)(args)
