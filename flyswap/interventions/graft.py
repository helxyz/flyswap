"""Connectome grafts: replacing one cell type's wiring with the other sex's.

The question a graft asks is sharper than the question silencing asks.
Silencing a dimorphic cell type asks "is this type necessary for the male
response?". Grafting asks "is this type's *female wiring* enough to make the
male brain behave like a female one?" — a sufficiency claim, and the one that
identifies which types carry a sex difference rather than merely participate
in it.

Why grafts happen at type level
-------------------------------
There is no neuron-to-neuron correspondence between two connectomes. A cell
type may have 6 neurons in the male and 9 in the female, and no fact of the
matter about which male neuron matches which female one. What *is* matched is
the type. So a graft rewrites the connection weights between type pairs,
leaving the host's own neurons and their counts untouched.

Concretely, grafting donor type T into a host means: for every pair (T, X),
rescale the host's T-to-X and X-to-T weights so the *type-level* input shares
match the donor's. Individual neurons keep their identity; what changes is how
much influence the type exerts and receives.

Controls built into this module
-------------------------------
A graft effect means nothing on its own, so the module also provides:

* :func:`null_graft` — swap an isomorphic type between two *same-sex* animals.
  Any effect here is individual variation, not sex.
* :func:`shuffle_labels` — randomise which types are called dimorphic, so the
  real labels can be shown to explain more than random ones.
* :func:`delete_type` — the silencing comparison, for types that exist in only
  one sex and cannot be grafted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from flyswap.graph.build import SignedGraph, type_level_matrix


@dataclass
class GraftSpec:
    """What was grafted, from where, into where."""

    cell_type: str
    donor: str
    host: str
    direction: str  # "both", "inputs", "outputs"
    kind: str = "graft"  # graft | null_graft | deletion
    notes: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "cell_type": self.cell_type,
            "donor": self.donor,
            "host": self.host,
            "direction": self.direction,
            "kind": self.kind,
            **self.notes,
        }


class TypeSpace:
    """Type-level views of a set of graphs, computed once and reused.

    Building a type-level matrix means a sparse triple product, which is slow
    to repeat for every one of thousands of grafts. This caches per graph.
    """

    def __init__(self, graphs: dict[str, SignedGraph]):
        self.graphs = graphs
        self._cache: dict[str, tuple[sparse.csr_matrix, list[str]]] = {}

    def matrix(self, name: str) -> tuple[sparse.csr_matrix, list[str]]:
        if name not in self._cache:
            self._cache[name] = type_level_matrix(self.graphs[name])
        return self._cache[name]

    def shared_types(self, *names: str) -> list[str]:
        """Types present in every named graph — the only ones graftable."""
        sets = [set(self.graphs[n].cell_types()) for n in names]
        return sorted(set.intersection(*sets))

    def type_lookup(self, name: str) -> dict[str, int]:
        _, types = self.matrix(name)
        return {t: i for i, t in enumerate(types)}


def _scale_block(
    W: sparse.lil_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    factor: float,
) -> int:
    """Multiply the (rows x cols) block by `factor`. Returns entries touched."""
    if len(rows) == 0 or len(cols) == 0 or factor == 1.0:
        return 0
    block = W[np.ix_(rows, cols)]
    touched = block.nnz
    if touched:
        W[np.ix_(rows, cols)] = block * factor
    return touched


def graft_type(
    host: SignedGraph,
    donor: SignedGraph,
    cell_type: str,
    space: TypeSpace,
    host_name: str,
    donor_name: str,
    direction: str = "both",
    max_factor: float = 50.0,
) -> tuple[SignedGraph, GraftSpec]:
    """Replace `cell_type`'s connection strengths in `host` with `donor`'s.

    For each partner type X, the host's type-level weight T_host[X, cell_type]
    is rescaled toward the donor's T_donor[X, cell_type]. The ratio is applied
    uniformly to the neuron-level entries in that block, which preserves the
    host's own neuron counts and sparsity while changing the type's influence.

    `max_factor` caps the rescaling. A donor weight far larger than a host
    weight near zero would otherwise produce an enormous factor driven by
    numerical noise rather than biology.
    """
    if direction not in ("both", "inputs", "outputs"):
        raise ValueError(f"direction must be both/inputs/outputs, got {direction!r}")

    T_host, types_host = space.matrix(host_name)
    T_donor, types_donor = space.matrix(donor_name)
    idx_host = space.type_lookup(host_name)
    idx_donor = space.type_lookup(donor_name)

    if cell_type not in idx_host:
        raise KeyError(f"{cell_type!r} absent from host {host_name}")
    if cell_type not in idx_donor:
        raise KeyError(f"{cell_type!r} absent from donor {donor_name}")

    th = idx_host[cell_type]
    td = idx_donor[cell_type]
    shared = [t for t in types_host if t in idx_donor]

    W = host.W.tolil(copy=True)
    target_rows = host.rows_for_type(cell_type)
    changed = 0
    factors: dict[str, dict[str, float]] = {"outputs": {}, "inputs": {}}

    eps = 1e-9
    for partner in shared:
        ph, pd_ = idx_host[partner], idx_donor[partner]
        partner_rows = host.rows_for_type(partner)

        # Outputs: cell_type -> partner. Row = postsynaptic = partner.
        if direction in ("both", "outputs"):
            h = float(T_host[ph, th])
            d = float(T_donor[pd_, td])
            if abs(h) > eps:
                factor = float(np.clip(d / h, -max_factor, max_factor))
                n = _scale_block(W, partner_rows, target_rows, factor)
                if n:
                    changed += n
                    factors["outputs"][partner] = round(factor, 4)

        # Inputs: partner -> cell_type. Row = postsynaptic = cell_type.
        if direction in ("both", "inputs"):
            h = float(T_host[th, ph])
            d = float(T_donor[td, pd_])
            if abs(h) > eps:
                factor = float(np.clip(d / h, -max_factor, max_factor))
                n = _scale_block(W, target_rows, partner_rows, factor)
                if n:
                    changed += n
                    factors["inputs"][partner] = round(factor, 4)

    spec = GraftSpec(
        cell_type=cell_type,
        donor=donor_name,
        host=host_name,
        direction=direction,
        kind="graft",
        notes={
            "entries_changed": changed,
            "n_host_neurons": int(len(target_rows)),
            "n_donor_neurons": int(len(donor.rows_for_type(cell_type))),
            "factors": factors,
        },
    )
    out = host.copy_with(
        W.tocsr(), f"graft({cell_type}, {donor_name}->{host_name}, {direction})"
    )
    out.config["graft"] = spec.describe()
    return out, spec


def null_graft(
    host: SignedGraph,
    donor: SignedGraph,
    cell_type: str,
    space: TypeSpace,
    host_name: str,
    donor_name: str,
    direction: str = "both",
) -> tuple[SignedGraph, GraftSpec]:
    """A graft between two same-sex animals.

    Mechanically identical to :func:`graft_type`; the difference is what it
    means. Donor and host are the same sex, so any readout change measures
    individual variation and reconstruction noise. A real dimorphic graft has
    to beat this to count.
    """
    if host.sex != donor.sex:
        raise ValueError(
            f"null graft needs same-sex donor and host, got {host.sex} and {donor.sex}"
        )
    grafted, spec = graft_type(
        host, donor, cell_type, space, host_name, donor_name, direction
    )
    spec = GraftSpec(
        cell_type=spec.cell_type,
        donor=spec.donor,
        host=spec.host,
        direction=spec.direction,
        kind="null_graft",
        notes=spec.notes,
    )
    grafted.config["graft"] = spec.describe()
    return grafted, spec


def delete_type(
    host: SignedGraph, cell_type: str, host_name: str = "host"
) -> tuple[SignedGraph, GraftSpec]:
    """Silence a cell type by zeroing its outgoing weights.

    The comparison condition for grafts, and the only option for types that
    exist in one sex only: a male-specific type has no female version to graft
    in, so deletion is the nearest counterfactual. Incoming weights are left
    alone, matching how optogenetic silencing works — the neuron still
    receives input, it just stops influencing its targets.
    """
    rows = host.rows_for_type(cell_type)
    W = host.W.tolil(copy=True)
    changed = 0
    if len(rows):
        block = W[:, rows]
        changed = block.nnz
        W[:, rows] = 0.0

    spec = GraftSpec(
        cell_type=cell_type,
        donor="none",
        host=host_name,
        direction="outputs",
        kind="deletion",
        notes={"entries_changed": int(changed), "n_neurons": int(len(rows))},
    )
    out = host.copy_with(W.tocsr(), f"delete({cell_type})")
    out.config["graft"] = spec.describe()
    return out, spec


def shuffle_labels(
    labels: dict[str, str], seed: int, keep: tuple[str, ...] = ()
) -> dict[str, str]:
    """Randomly reassign dimorphism labels across cell types.

    Control for the claim "dimorphic types matter more". If grafting a
    randomly relabelled set of types explains as much as grafting the real
    dimorphic set, the labels were not doing any work.
    """
    rng = np.random.default_rng(seed)
    movable = {k: v for k, v in labels.items() if k not in keep}
    keys = list(movable)
    values = list(movable.values())
    rng.shuffle(values)
    out = dict(labels)
    out.update(dict(zip(keys, values, strict=True)))
    return out
