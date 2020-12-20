import argparse
import torch

from common.dgl_datasets import KarateClub
from embeddings.gcn_embedding import GCNEmbedding

if __name__ == "__main__":
    # Parsing arguments.
    parser = argparse.ArgumentParser(
        description="Graphs in Space: Graph Embeddings for Machine Learning on Complex Data -- Evaluation."
    )
    parser.add_argument("-g", "--graph", help="Path to the graph, or name of the dataset from the dataset pool (e.g. "
                                              "'KarateClub')", required=True)
    args = parser.parse_args()

    # Do something...
    if args.graph == "KarateClub":
        g = KarateClub().load()
        dimension = 5
        embedding = GCNEmbedding(g, dimension)
        embedding.embed(args={
            "labeled_nodes": torch.tensor([0, 33]),
            "labels": torch.tensor([0, 1]),
            "epochs": 50
        })
        for i in range(34):
            print(f"Embedding of node {i}: {embedding[i].detach().numpy()}")
