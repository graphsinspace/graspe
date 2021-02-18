import statistics

from common.graph_loaders import load_npz


def test_npz():
    g = load_npz("../../data/citeseer.npz")
    print(g)
    print(g.nodes())
    print(g.edges())
    nx = g.to_networkx()
    in_dg = [x[1] for x in nx.in_degree(nx.nodes)]
    out_dg = [x[1] for x in nx.out_degree(nx.nodes)]
    print(
        "In degrees: std={}, min={}, max={}".format(
            statistics.stdev(in_dg), min(in_dg), max(in_dg)
        )
    )
    print(
        "Out degrees: std={}, min={}, max={}".format(
            statistics.stdev(out_dg), min(out_dg), max(out_dg)
        )
    )

    assert g
