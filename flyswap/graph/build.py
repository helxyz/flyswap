"""Signed, normalised connectivity matrices.

The central object is :class:`SignedGraph`: a sparse matrix W where W[i, j] is
the influence of neuron j on neuron i, already signed and normalised, plus the
bookkeeping needed to interpret rows (cell types, sides, superclasses).

Why input fraction rather than raw synapse counts
-------------------------------------------------
Published work reports that a male neuron carries roughly 1.8x the synapses of
its female counterpart across thousands of matched cell types — a difference
in how the two volumes were reconstructed, not in the animals. Comparing raw
counts across datasets would therefore measure the microscope, not the brain.

Dividing each neuron's incoming weights by its total incoming synapses gives
the *share* of a neuron's input that arrives from each partner. That quantity
is dimensionless and cancels a uniform density difference exactly. It is also
the quantity most comparable to what a neuron actually integrates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from flyswap.data.schema import Connectome, config_digest
from flyswap.graph.signs import SignRule, get_rule

Normalisation = Literal["input_fraction", "raw", "log"]


@dataclass
class SignedGraph:
    """A connectome as a signed sparse matrix plus neuron metadata.

    Attributes
    ----------
    W : scipy.sparse.csr_matrix
        Shape (n, n). W[i, j] is the signed influence of j on i, so a
        simulation step is ``W @ r``. Note this is the transpose of the raw
        edge list's natural orientation; the conversion happens once, here.
    neurons : pandas.DataFrame
        Row i of this frame describes neuron i, in matrix order.
    signs : numpy.ndarray
        Per-neuron sign (+1/-1/0) actually used, after any perturbation.
    """

    W: sparse.csr_matrix
    neurons: pd.DataFrame
    signs: np.ndarray
    dataset: str
    sex: str
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.neurons)
        if self.W.shape != (n, n):
            raise ValueError(f"W shape {self.W.shape} does not match {n} neurons")
        self._type_index: dict[str, np.ndarray] | None = None

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def nnz(self) -> int:
        return self.W.nnz

    @property
    def config_hash(self) -> str:
        return config_digest(self.config)

    def rows_for_type(self, cell_type: str) -> np.ndarray:
        """Matrix rows belonging to a cell type (cached)."""
        if self._type_index is None:
            self._type_index = {
                str(t): np.flatnonzero((self.neurons["cell_type"] == t).to_numpy())
                for t in self.neurons["cell_type"].unique()
            }
        return self._type_index.get(cell_type, np.array([], dtype=np.int64))

    def rows_for_types(self, cell_types: list[str]) -> np.ndarray:
        if not cell_types:
            return np.array([], dtype=np.int64)
        parts = [self.rows_for_type(t) for t in cell_types]
        return np.concatenate([p for p in parts if len(p)]) if any(
            len(p) for p in parts
        ) else np.array([], dtype=np.int64)

    def cell_types(self) -> list[str]:
        return sorted(str(t) for t in self.neurons["cell_type"].unique())

    def type_codes(self) -> tuple[np.ndarray, list[str]]:
        """Integer code per neuron plus the ordered type names.

        Used by the attribution layer, which needs one gain parameter per cell
        type and a way to scatter it back onto neurons.
        """
        types = self.cell_types()
        lookup = {t: i for i, t in enumerate(types)}
        codes = self.neurons["cell_type"].map(lambda t: lookup[str(t)]).to_numpy()
        return codes.astype(np.int32), types

    def copy_with(self, W: sparse.csr_matrix, note: str) -> SignedGraph:
        """A new graph sharing metadata but carrying a modified matrix."""
        config = dict(self.config)
        config.setdefault("derivations", [])
        config["derivations"] = [*config["derivations"], note]
        return SignedGraph(
            W=W.tocsr(),
            neurons=self.neurons,
            signs=self.signs,
            dataset=self.dataset,
            sex=self.sex,
            config=config,
        )

    def __repr__(self) -> str:
        return (
            f"SignedGraph({self.dataset}, sex={self.sex}, n={self.n:,}, "
            f"nnz={self.nnz:,}, norm={self.config.get('normalisation')})"
        )


def build_graph(
    conn: Connectome,
    sign_rule: str | SignRule = "shiu2024",
    normalisation: Normalisation = "input_fraction",
    drop_zero_sign: bool = False,
    sign_perturbation: float = 0.0,
    sign_seed: int = 0,
) -> SignedGraph:
    """Build the signed matrix for a connectome.

    Parameters
    ----------
    normalisation
        ``input_fraction`` divides each row by that neuron's total incoming
        synapses (the default, and the only one comparable across datasets).
        ``raw`` keeps synapse counts. ``log`` uses log1p(count), a middle
        ground sometimes used to compress the six-order-of-magnitude range of
        real connection weights.
    drop_zero_sign
        If True, remove edges from neurons whose transmitter maps to 0 instead
        of keeping them as explicit zeros. Smaller matrix, same dynamics.
    sign_perturbation
        Fraction of signs to flip, for sensitivity runs. Recorded in config.
    """
    rule = get_rule(sign_rule) if isinstance(sign_rule, str) else sign_rule

    neurons = conn.neurons.reset_index(drop=True)
    index = {int(b): i for i, b in enumerate(neurons["body_id"])}
    n = len(neurons)

    signs = rule.apply(neurons["nt"])
    if sign_perturbation > 0:
        from flyswap.graph.signs import perturb

        signs = perturb(signs, sign_perturbation, sign_seed)

    pre = conn.edges["pre"].map(index).to_numpy(dtype=np.int64)
    post = conn.edges["post"].map(index).to_numpy(dtype=np.int64)
    count = conn.edges["count"].to_numpy(dtype=np.float64)

    # Orientation: rows are postsynaptic so that W @ r is "input to each
    # neuron". The edge list is (pre -> post), hence post as the row index.
    raw = sparse.coo_matrix((count, (post, pre)), shape=(n, n)).tocsr()

    if normalisation == "input_fraction":
        total_in = np.asarray(raw.sum(axis=1)).ravel()
        # A neuron with no input keeps a zero row; guard the division.
        scale = np.where(total_in > 0, 1.0 / np.maximum(total_in, 1e-12), 0.0)
        weights = sparse.diags(scale) @ raw
    elif normalisation == "raw":
        weights = raw
    elif normalisation == "log":
        weights = raw.copy()
        weights.data = np.log1p(weights.data)
    else:
        raise ValueError(f"unknown normalisation {normalisation!r}")

    # Apply the presynaptic sign along columns.
    W = (weights @ sparse.diags(signs.astype(np.float64))).tocsr()

    if drop_zero_sign:
        W.eliminate_zeros()

    config = {
        "dataset": conn.spec.name,
        "sex": conn.spec.sex,
        "normalisation": normalisation,
        "drop_zero_sign": drop_zero_sign,
        "sign_perturbation": sign_perturbation,
        "sign_seed": sign_seed if sign_perturbation else None,
        **rule.describe(),
        "n_neurons": n,
        "n_edges": conn.n_edges,
        "n_synapses": conn.n_synapses,
        "provenance": conn.provenance,
    }

    return SignedGraph(
        W=W.tocsr(),
        neurons=neurons,
        signs=signs,
        dataset=conn.spec.name,
        sex=conn.spec.sex,
        config=config,
    )


def type_level_matrix(graph: SignedGraph) -> tuple[sparse.csr_matrix, list[str]]:
    """Collapse the neuron graph to a type-by-type matrix.

    Each entry is the mean input fraction a neuron of the postsynaptic type
    receives from the presynaptic type. Means, not sums, because the two sexes
    have different neuron counts per type: summing would confound "more
    neurons" with "stronger connection".
    """
    codes, types = graph.type_codes()
    n_types = len(types)

    # Membership matrix M[t, i] = 1 if neuron i has type t.
    M = sparse.coo_matrix(
        (np.ones(graph.n), (codes, np.arange(graph.n))), shape=(n_types, graph.n)
    ).tocsr()

    counts = np.asarray(M.sum(axis=1)).ravel()
    inv = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)

    # Average over postsynaptic members, sum over presynaptic members:
    # a postsynaptic neuron's input share from a type is the total over that
    # type's members.
    T = sparse.diags(inv) @ M @ graph.W @ M.T
    return T.tocsr(), types


def degree_preserving_shuffle(
    graph: SignedGraph, seed: int, n_swaps_per_edge: int = 10
) -> SignedGraph:
    """Randomise wiring while keeping each neuron's in- and out-degree.

    This is the null model that makes a positive result meaningful. If a
    stimulus drives its known target just as well in a shuffled graph, the
    result came from degree statistics, not from the specific wiring.

    Implemented as double-edge swaps on the sparsity pattern: pick two edges
    (a->b) and (c->d), rewire to (a->d) and (c->b). Degrees are invariant by
    construction; weights travel with their edges.
    """
    rng = np.random.default_rng(seed)
    coo = graph.W.tocoo()
    rows = coo.row.copy()
    cols = coo.col.copy()
    data = coo.data.copy()
    m = len(data)
    if m < 2:
        return graph.copy_with(graph.W, f"shuffle(seed={seed}) [no-op]")

    existing = set(zip(rows.tolist(), cols.tolist(), strict=True))
    target = n_swaps_per_edge * m
    done = 0
    attempts = 0
    max_attempts = target * 10

    while done < target and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b = rows[i], cols[i]
        c, d = rows[j], cols[j]
        # Avoid self-loops and duplicate edges.
        if a == d or c == b:
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        existing.add((a, d))
        existing.add((c, b))
        cols[i], cols[j] = d, b
        done += 1

    W = sparse.coo_matrix((data, (rows, cols)), shape=graph.W.shape).tocsr()
    out = graph.copy_with(W, f"degree_preserving_shuffle(seed={seed}, swaps={done})")
    out.config["shuffle"] = {"seed": seed, "swaps": done, "attempts": attempts}
    return out
