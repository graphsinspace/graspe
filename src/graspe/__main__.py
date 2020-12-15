import argparse
from common.graph import Graph
from common.graph_loader import load_csv
from embeddings.embedding_example import EmbeddingExample

if __name__ == '__main__':
    # Parsing arguments.
    parser = argparse.ArgumentParser(description='Graphs in Space: Graph Embeddings for Machine Learning on Complex Data -- Evaluation.')
    parser.add_argument('-g', '--graph', help='Path to the graph.', required=True)
    args = parser.parse_args()

    # Do something...
    g = load_csv(args.graph)
    e = EmbeddingExample(g, 10)
