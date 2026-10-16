"""flyswap: counterfactual cross-sex simulation of the Drosophila connectome.

The package is organised in layers, each usable on its own:

    data/          dataset-agnostic schema + loaders (MaleCNS, FAFB, BANC)
    types/         common cell-type space across datasets
    graph/         signed, normalised connectivity matrices
    sim/           rate (differentiable) and spiking simulators
    interventions/ cross-sex grafts, deletions, null controls
    attribution/   gradient ranking of cell types
    experiments/   pre-registered protocols and runners
    bench/         throughput and scaling measurements

Nothing here bundles connectome data. Datasets are downloaded from their
public sources under their own licences (all CC-BY 4.0 as of 2026-10).
"""

__version__ = "0.1.0"

from flyswap.data.schema import Connectome, DatasetSpec  # noqa: F401
