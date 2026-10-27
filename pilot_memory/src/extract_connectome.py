"""
Pulls the central-complex navigation circuit (EPG, PEN1, PEN2, PFL, PFN,
plus ER ring neurons as the BU -> EB sensory input stage and Delta7 as the
PB inhibitory interneurons) from neuPrint's
male-cns:v1.0 dataset and saves two tables to data/:

  data/neurons.csv      one row per neuron (bodyId, type, and whatever
                         per-neuron fields the dataset exposes)
  data/connections.csv  one row per (pre, post) pair with total synapse
                         weight between them, summed across brain regions

Requires NEUPRINT_TOKEN to be set in the environment (get it from your
account page at neuprint.janelia.org after logging in).
"""
import os
import sys

from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

DATASET = "male-cns:v1.0"
SERVER = "neuprint.janelia.org"
# neuPrint names subtypes (e.g. PEN_a(PEN1), PFL3, PFNd, ER4d), so match
# each family by prefix. ER = ring neurons, the sensory input to the EB.
# Delta7 = glutamatergic (inhibitory) PB neurons the compass ring attractor needs.
CELL_TYPE_PATTERNS = ["EPG", "PEN", "PFL", "PFN", "ER", "Delta7"]
CELL_TYPE_REGEX = "|".join(f"{p}.*" for p in CELL_TYPE_PATTERNS)


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit("Set NEUPRINT_TOKEN in your environment first (see account page token).")

    # neuprint only keeps a weakref to the default client, so hold on to it.
    client = Client(SERVER, dataset=DATASET, token=token)

    criteria = NC(type=CELL_TYPE_REGEX, regex=True)

    print(f"Fetching neurons matching {CELL_TYPE_REGEX!r} ...")
    neuron_df, roi_counts_df = fetch_neurons(criteria, client=client)
    print(f"  -> {len(neuron_df)} neurons found")
    if neuron_df.empty:
        sys.exit("No neurons matched; check the type names in this dataset.")
    print("  -> neurons per type:")
    print(neuron_df["type"].value_counts().sort_index().to_string())
    print(f"  -> columns available: {list(neuron_df.columns)}")

    print("Fetching synaptic connections among them ...")
    _, conn_df = fetch_adjacencies(sources=criteria, targets=criteria, client=client)
    print(f"  -> {len(conn_df)} raw (pre, post, roi) rows")
    print(f"  -> columns available: {list(conn_df.columns)}")

    # Collapse per-ROI rows into one total weight per (pre, post) pair.
    total_conn_df = (
        conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()
    )
    print(f"  -> {len(total_conn_df)} unique (pre, post) pairs after collapsing ROIs")

    os.makedirs("data", exist_ok=True)
    neuron_df.to_csv("data/neurons.csv", index=False)
    total_conn_df.to_csv("data/connections.csv", index=False)
    print("Saved data/neurons.csv and data/connections.csv")


if __name__ == "__main__":
    main()
