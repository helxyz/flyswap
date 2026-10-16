"""Dataset-specific readers that produce a :class:`Connectome`.

Each loader takes a directory of files downloaded from that dataset's official
source and returns the shared schema. The column mappings below follow the
public release layouts as of October 2026; releases do change, so each loader
checks for its expected columns and raises a clear error naming the file
rather than failing deep inside a merge.

No loader downloads anything. Fetching is a deliberate manual step: the files
are large, the licences require attribution, and pinning a local copy with a
recorded hash is what makes a result reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flyswap.data.schema import (
    DATASETS,
    Connectome,
    file_digest,
    normalise_nt,
)


class MissingDataError(FileNotFoundError):
    """Raised when a dataset file is absent, with the download URL attached."""

    def __init__(self, dataset: str, path: Path, expected: list[str]):
        spec = DATASETS[dataset]
        super().__init__(
            f"{dataset}: expected {expected} in {path}.\n"
            f"Download from {spec.download}\n"
            f"Cite: {spec.citation}"
        )


def _require(path: Path, candidates: list[str], dataset: str) -> Path:
    """First existing candidate filename, else a helpful error."""
    for name in candidates:
        hit = path / name
        if hit.exists():
            return hit
    raise MissingDataError(dataset, path, candidates)


def _finalise(
    dataset: str,
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    provenance: dict,
    min_syn: int = 1,
) -> Connectome:
    """Shared tail of every loader: type coercion, filtering, validation."""
    neurons = neurons.copy()
    neurons["body_id"] = neurons["body_id"].astype("int64")
    for col in ("cell_type", "side", "superclass", "fru_dsx"):
        neurons[col] = neurons[col].fillna("unknown").astype("string")
    neurons["nt"] = neurons["nt"].map(normalise_nt).astype("string")
    neurons["nt_conf"] = neurons["nt_conf"].fillna(0.0).astype("float32")

    edges = edges.copy()
    edges["pre"] = edges["pre"].astype("int64")
    edges["post"] = edges["post"].astype("int64")
    edges["count"] = edges["count"].astype("int32")

    # Aggregate duplicate pairs before thresholding, so a pair split across
    # rows is not discarded piecemeal.
    edges = edges.groupby(["pre", "post"], as_index=False)["count"].sum()
    before = len(edges)
    edges = edges[edges["count"] >= min_syn]

    # Keep only edges whose endpoints survived neuron filtering.
    keep = set(neurons["body_id"].astype("int64"))
    mask = edges["pre"].isin(keep) & edges["post"].isin(keep)
    dropped_endpoints = int((~mask).sum())
    edges = edges[mask]

    provenance = dict(provenance)
    provenance.update(
        {
            "min_syn": min_syn,
            "edges_before_threshold": before,
            "edges_dropped_unknown_endpoint": dropped_endpoints,
        }
    )

    conn = Connectome(
        spec=DATASETS[dataset],
        neurons=neurons[
            ["body_id", "cell_type", "side", "superclass", "nt", "nt_conf", "fru_dsx"]
        ],
        edges=edges[["pre", "post", "count"]],
        provenance=provenance,
    )
    conn.validate()
    return conn


def load_malecns(
    data_dir: Path | str,
    min_syn: int = 1,
    status: tuple[str, ...] = ("Traced",),
) -> Connectome:
    """MaleCNS v1.0 from the Janelia release.

    Expects the three Feather tables, renamed to short names:
      connectome-weights.feather, body-annotations.feather,
      body-neurotransmitters.feather
    """
    data_dir = Path(data_dir)
    w_path = _require(data_dir, ["connectome-weights.feather"], "malecns")
    a_path = _require(data_dir, ["body-annotations.feather"], "malecns")
    nt_path = _require(data_dir, ["body-neurotransmitters.feather"], "malecns")

    ann = pd.read_feather(a_path)
    nt = pd.read_feather(nt_path)
    weights = pd.read_feather(w_path)

    ann = _rename_first_match(
        ann,
        {
            "body_id": ["bodyId", "body_id", "bodyid"],
            "cell_type": ["type", "cell_type", "cellType"],
            "side": ["side", "soma_side", "somaSide"],
            "superclass": ["superclass", "class", "super_class"],
            "status": ["status", "statusLabel"],
            "fru_dsx": ["fru_dsx", "fruDsx", "dimorphism_gene"],
        },
        dataset="malecns",
        required=["body_id"],
    )
    if "status" in ann.columns and status:
        ann = ann[ann["status"].isin(status)]

    nt = _rename_first_match(
        nt,
        {
            "body_id": ["bodyId", "body_id", "bodyid"],
            "nt": ["consensus_nt", "predictedNt", "nt", "neurotransmitter"],
            "nt_conf": ["consensus_nt_conf", "predictedNtProb", "nt_conf", "confidence"],
        },
        dataset="malecns",
        required=["body_id"],
    )

    weights = _rename_first_match(
        weights,
        {
            "pre": ["bodyId_pre", "pre", "pre_id", "source"],
            "post": ["bodyId_post", "post", "post_id", "target"],
            "count": ["weight", "count", "syn_count", "n_syn"],
        },
        dataset="malecns",
        required=["pre", "post", "count"],
    )

    neurons = ann.merge(nt[["body_id", "nt", "nt_conf"]], on="body_id", how="left")
    neurons = _ensure_columns(neurons)

    provenance = {
        "files": {
            p.name: file_digest(p) for p in (w_path, a_path, nt_path)
        },
        "status_filter": list(status),
    }
    return _finalise("malecns", neurons, weights, provenance, min_syn)


def load_fafb(data_dir: Path | str, min_syn: int = 5) -> Connectome:
    """FlyWire FAFB v783 from Codex downloads.

    Codex's public connections export is already thresholded at 5 synapses,
    which is why `min_syn` defaults to 5 here and 1 for MaleCNS. Any
    cross-dataset comparison must set the same value explicitly.
    """
    data_dir = Path(data_dir)
    n_path = _require(data_dir, ["neurons.csv.gz", "neurons.csv"], "fafb")
    c_path = _require(
        data_dir,
        ["connections_princeton.csv.gz", "connections.csv.gz", "connections.csv"],
        "fafb",
    )
    cls_path = _require(
        data_dir, ["classification.csv.gz", "classification.csv"], "fafb"
    )

    neurons = pd.read_csv(n_path)
    classification = pd.read_csv(cls_path)
    conns = pd.read_csv(c_path)

    neurons = _rename_first_match(
        neurons,
        {
            "body_id": ["root_id", "body_id", "id"],
            "cell_type": ["cell_type", "type", "hemibrain_type"],
            "side": ["side", "soma_side"],
            "nt": ["nt_type", "nt", "neurotransmitter"],
            "nt_conf": ["nt_type_score", "nt_conf", "confidence"],
        },
        dataset="fafb",
        required=["body_id"],
    )
    classification = _rename_first_match(
        classification,
        {
            "body_id": ["root_id", "body_id", "id"],
            "superclass": ["super_class", "superclass", "class"],
            "cls_cell_type": ["cell_type", "type"],
            "cls_side": ["side"],
        },
        dataset="fafb",
        required=["body_id"],
    )
    conns = _rename_first_match(
        conns,
        {
            "pre": ["pre_root_id", "pre", "pre_pt_root_id"],
            "post": ["post_root_id", "post", "post_pt_root_id"],
            "count": ["syn_count", "count", "weight"],
        },
        dataset="fafb",
        required=["pre", "post", "count"],
    )

    merge_cols = ["body_id"] + [
        c for c in ("superclass", "cls_cell_type", "cls_side") if c in classification
    ]
    neurons = neurons.merge(classification[merge_cols], on="body_id", how="left")
    # Classification is the better-curated source for type and side.
    if "cls_cell_type" in neurons:
        neurons["cell_type"] = neurons["cls_cell_type"].fillna(
            neurons.get("cell_type", pd.Series(dtype="object"))
        )
    if "cls_side" in neurons:
        neurons["side"] = neurons["cls_side"].fillna(
            neurons.get("side", pd.Series(dtype="object"))
        )
    neurons = _ensure_columns(neurons)

    provenance = {
        "files": {p.name: file_digest(p) for p in (n_path, c_path, cls_path)},
        "note": "Codex connections export is pre-thresholded at 5 synapses.",
    }
    return _finalise("fafb", neurons, conns, provenance, min_syn)


def load_banc(data_dir: Path | str, min_syn: int = 1) -> Connectome:
    """BANC v888 (female brain + nerve cord) from the static data archive."""
    data_dir = Path(data_dir)
    e_path = _require(
        data_dir, ["banc_888_edgelist_simple_v3.feather", "edges.feather"], "banc"
    )
    m_path = _require(data_dir, ["banc_888_meta.feather", "meta.feather"], "banc")

    meta = pd.read_feather(m_path)
    edges = pd.read_feather(e_path)

    meta = _rename_first_match(
        meta,
        {
            "body_id": ["root_id", "body_id", "id", "pt_root_id"],
            "cell_type": ["cell_type", "type", "malecns_type"],
            "side": ["side", "soma_side"],
            "superclass": ["super_class", "superclass", "class"],
            "nt": ["nt", "top_nt", "neurotransmitter"],
            "nt_conf": ["nt_conf", "top_nt_conf", "confidence"],
            "status": ["status", "proofread_status"],
        },
        dataset="banc",
        required=["body_id"],
    )
    edges = _rename_first_match(
        edges,
        {
            "pre": ["pre", "pre_root_id", "pre_pt_root_id", "source"],
            "post": ["post", "post_root_id", "post_pt_root_id", "target"],
            "count": ["count", "syn_count", "weight"],
        },
        dataset="banc",
        required=["pre", "post", "count"],
    )

    if "superclass" in meta.columns:
        meta = meta[~meta["superclass"].isin(["glia", "not_a_neuron"])]
    meta = _ensure_columns(meta)

    provenance = {"files": {p.name: file_digest(p) for p in (e_path, m_path)}}
    return _finalise("banc", meta, edges, provenance, min_syn)


# -- helpers --------------------------------------------------------------


def _rename_first_match(
    df: pd.DataFrame,
    mapping: dict[str, list[str]],
    dataset: str,
    required: list[str],
) -> pd.DataFrame:
    """Rename whichever candidate column is present to the canonical name.

    Releases rename columns between versions, so each canonical name carries a
    list of spellings seen in the wild. Missing optional columns are fine;
    missing required ones raise immediately with the available columns listed,
    which is far easier to debug than a KeyError three merges later.
    """
    out = df.copy()
    renames: dict[str, str] = {}
    for canonical, candidates in mapping.items():
        if canonical in out.columns:
            continue
        for cand in candidates:
            if cand in out.columns:
                renames[cand] = canonical
                break
    out = out.rename(columns=renames)

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"{dataset}: could not find columns {missing}. "
            f"Available: {sorted(out.columns)[:25]}"
        )
    return out


def _ensure_columns(neurons: pd.DataFrame) -> pd.DataFrame:
    """Fill in schema columns a dataset does not provide."""
    out = neurons.copy()
    defaults = {
        "cell_type": "unknown",
        "side": "U",
        "superclass": "unknown",
        "nt": "unknown",
        "nt_conf": np.float32(0.0),
        "fru_dsx": "none",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out
