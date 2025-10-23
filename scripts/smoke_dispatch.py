from engine.data_loaders.transmission import load_edges
from engine.data_loaders.units import load_unit_fleet
try:
    from engine.orchestrate import run_policy_simulation
except Exception:
    run_policy_simulation = None

def main():
    edges = load_edges()
    units = load_unit_fleet()
    assert not edges.empty, "edges empty"
    assert not units.empty, "units empty"
    if run_policy_simulation:
        config = {"years":[2025], "policy":{"type":"price","start_price":0.0}}
        inputs = {"edges":edges, "units":units}
        res = run_policy_simulation(config, inputs)
        print("run ok:", bool(res))
    else:
        print("Loaded edges", edges.shape, "units", units.shape)

if __name__ == "__main__":
    main()
