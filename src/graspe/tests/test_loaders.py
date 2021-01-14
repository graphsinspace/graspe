from common.graph_loaders import load_npz


def test_npz():
    g = load_npz("../../data/cora.npz")
    print(g)
    print(g.nodes())
    print(g.edges())
    assert g
