from common.dataset_pool import DatasetPool
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_gcn import GCNEmbedding


def test_gae_deterministic():
    embeddings = []
    for _ in range(5):
        g = DatasetPool.load("karate_club_graph")
        gae_embedding = GAEEmbedding(
            g, d=10, epochs=5, variational=False, linear=False, deterministic=True
        )
        gae_embedding.embed()
        assert gae_embedding._embedding is not None
        embeddings.append(gae_embedding._embedding)

    first = embeddings[0]

    for e in embeddings[1:]:
        for emb_node1, emb_node2 in zip(first.values(), e.values()):
            assert all(abs(emb_node1 - emb_node2) <= 1e-6)


def test_gcn_deterministic():
    embeddings = []
    for _ in range(5):
        g = DatasetPool.load("karate_club_graph")
        gae_embedding = GCNEmbedding(g, d=10, epochs=5, deterministic=True)
        gae_embedding.embed()
        assert gae_embedding._embedding is not None
        embeddings.append(gae_embedding._embedding)

    first = embeddings[0]

    for e in embeddings[1:]:
        for emb_node1, emb_node2 in zip(first.values(), e.values()):
            assert all(abs(emb_node1 - emb_node2) <= 1e-6)


if __name__ == "__main__":
    test_gae_deterministic()
    test_gcn_deterministic()
