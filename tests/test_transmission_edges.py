from engine.io.transmission import load_ei_edges
from engine.network.topology import build_transmission_caps


def test_edges_no_inferred_pairs():
    edges = load_ei_edges()
    zones = sorted(set(edges["from_region"]) | set(edges["to_region"]))
    mat, caps = build_transmission_caps(edges, zones)

    for a in zones:
        for b in zones:
            if (a, b) not in caps:
                assert mat.at[a, b] == 0.0
