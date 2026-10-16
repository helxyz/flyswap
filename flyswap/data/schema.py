"""One schema for three connectomes.

MaleCNS, FlyWire FAFB and BANC ship different column names, different id
conventions and different neurotransmitter vocabularies. Everything downstream
of this module sees a single shape:

    neurons: body_id, cell_type, side, superclass, nt, nt_conf, fru_dsx, dataset
    edges:   pre, post, count

`Connectome` holds those two frames plus the provenance needed to reproduce
them. It deliberately does NOT hold a matrix: building weights is the graph
layer's job, and keeping them separate means one ingested dataset can be
turned into several weightings (raw counts, input fraction, type-level).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Sex = Literal["male", "female"]

NEURON_COLUMNS = {
    "body_id": "int64",
    "cell_type": "string",
    "side": "string",  # L, R, M (midline), or U (unknown)
    "superclass": "string",  # sensory, intrinsic, descending, motor, ...
    "nt": "string",  # acetylcholine, gaba, glutamate, ...
    "nt_conf": "float32",
    "fru_dsx": "string",  # fru, dsx, fru+dsx, or none/unknown
}

EDGE_COLUMNS = {
    "pre": "int64",
    "post": "int64",
    "count": "int32",
}

#: Transmitter vocabulary after normalisation. Dataset-specific spellings
#: ("ACh", "acetylcholine", "ach") all collapse to these.
TRANSMITTERS = (
    "acetylcholine",
    "gaba",
    "glutamate",
    "histamine",
    "dopamine",
    "serotonin",
    "octopamine",
    "unknown",
)

_NT_ALIASES = {
    "ach": "acetylcholine",
    "acetylcholine": "acetylcholine",
    "cholinergic": "acetylcholine",
    "gaba": "gaba",
    "gabaergic": "gaba",
    "glut": "glutamate",
    "glutamate": "glutamate",
    "glutamatergic": "glutamate",
    "hist": "histamine",
    "histamine": "histamine",
    "da": "dopamine",
    "dopamine": "dopamine",
    "5ht": "serotonin",
    "serotonin": "serotonin",
    "oct": "octopamine",
    "octopamine": "octopamine",
    "unk": "unknown",
    "unclear": "unknown",
    "": "unknown",
}


def normalise_nt(value: object) -> str:
    """Map a dataset's transmitter spelling onto the shared vocabulary."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    key = str(value).strip().lower()
    return _NT_ALIASES.get(key, "unknown")


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset comes from and what it covers.

    `regions` matters for comparisons: FAFB is brain-only, so a male-vs-FAFB
    contrast must be restricted to brain neurons, while male-vs-BANC can
    include the nerve cord. Experiments read this rather than hard-coding it.
    """

    name: str
    sex: Sex
    regions: tuple[str, ...]  # ("brain",) or ("brain", "vnc")
    citation: str
    download: str

    @property
    def has_vnc(self) -> bool:
        return "vnc" in self.regions


DATASETS: dict[str, DatasetSpec] = {
    "malecns": DatasetSpec(
        name="malecns",
        sex="male",
        regions=("brain", "vnc"),
        citation=(
            "Berg, S. et al. Sexual dimorphism in the complete Drosophila male "
            "central nervous system connectome. Cell 189, 5504-5526 (2026). "
            "doi:10.1016/j.cell.2026.08.015. CC-BY 4.0."
        ),
        download="https://male-cns.janelia.org/download/",
    ),
    "fafb": DatasetSpec(
        name="fafb",
        sex="female",
        regions=("brain",),
        citation=(
            "Dorkenwald et al. (2024); Schlegel et al. (2024); Matsliah et al. "
            "(2024). FlyWire FAFB v783. CC-BY 4.0."
        ),
        download="https://codex.flywire.ai/api/download?dataset=fafb",
    ),
    "banc": DatasetSpec(
        name="banc",
        sex="female",
        regions=("brain", "vnc"),
        citation="Bates et al. (2025) bioRxiv. BANC v888. CC-BY 4.0.",
        download="https://codex.flywire.ai/banc",
    ),
}


@dataclass
class Connectome:
    """A dataset in the shared schema, plus how it was produced.

    Invariants (checked by :meth:`validate`):
      * every edge endpoint exists in `neurons`
      * body_id is unique
      * counts are positive
      * nt values are in TRANSMITTERS
    """

    spec: DatasetSpec
    neurons: pd.DataFrame
    edges: pd.DataFrame
    provenance: dict = field(default_factory=dict)

    # -- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        self.neurons = self.neurons.reset_index(drop=True)
        self.edges = self.edges.reset_index(drop=True)
        self._index: dict[int, int] | None = None

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_synapses(self) -> int:
        return int(self.edges["count"].sum())

    def index_of(self, body_ids: np.ndarray | list[int]) -> np.ndarray:
        """Row positions for body ids, for indexing into matrices."""
        if self._index is None:
            self._index = {int(b): i for i, b in enumerate(self.neurons["body_id"])}
        return np.array([self._index[int(b)] for b in body_ids], dtype=np.int64)

    def select(
        self,
        cell_type: str | None = None,
        type_regex: str | None = None,
        side: str | None = None,
        superclass: str | None = None,
    ) -> np.ndarray:
        """Row positions of neurons matching every supplied filter (AND)."""
        mask = np.ones(len(self.neurons), dtype=bool)
        if cell_type is not None:
            mask &= (self.neurons["cell_type"] == cell_type).to_numpy()
        if type_regex is not None:
            matched = self.neurons["cell_type"].str.match(type_regex, na=False)
            mask &= matched.to_numpy()
        if side is not None:
            mask &= (self.neurons["side"] == side).to_numpy()
        if superclass is not None:
            mask &= (self.neurons["superclass"] == superclass).to_numpy()
        return np.flatnonzero(mask)

    # -- validation -------------------------------------------------------

    def validate(self) -> None:
        missing = set(NEURON_COLUMNS) - set(self.neurons.columns)
        if missing:
            raise ValueError(f"neurons missing columns: {sorted(missing)}")
        missing = set(EDGE_COLUMNS) - set(self.edges.columns)
        if missing:
            raise ValueError(f"edges missing columns: {sorted(missing)}")

        if self.neurons["body_id"].duplicated().any():
            dupes = self.neurons.loc[self.neurons["body_id"].duplicated(), "body_id"]
            raise ValueError(f"duplicate body_id values, e.g. {dupes.iloc[0]}")

        known = set(self.neurons["body_id"].astype("int64"))
        for col in ("pre", "post"):
            unknown = set(self.edges[col].astype("int64")) - known
            if unknown:
                raise ValueError(
                    f"{len(unknown)} edge {col} ids not in neurons, e.g. {next(iter(unknown))}"
                )

        if (self.edges["count"] <= 0).any():
            raise ValueError("edge counts must be positive")

        bad_nt = set(self.neurons["nt"].dropna()) - set(TRANSMITTERS)
        if bad_nt:
            raise ValueError(f"unknown transmitter labels: {sorted(bad_nt)}")

    # -- persistence ------------------------------------------------------

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.neurons.to_parquet(directory / "neurons.parquet", index=False)
        self.edges.to_parquet(directory / "edges.parquet", index=False)
        meta = {
            "spec": {
                "name": self.spec.name,
                "sex": self.spec.sex,
                "regions": list(self.spec.regions),
                "citation": self.spec.citation,
                "download": self.spec.download,
            },
            "provenance": self.provenance,
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "n_synapses": self.n_synapses,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> Connectome:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        spec_d = meta["spec"]
        spec = DatasetSpec(
            name=spec_d["name"],
            sex=spec_d["sex"],
            regions=tuple(spec_d["regions"]),
            citation=spec_d["citation"],
            download=spec_d["download"],
        )
        return cls(
            spec=spec,
            neurons=pd.read_parquet(directory / "neurons.parquet"),
            edges=pd.read_parquet(directory / "edges.parquet"),
            provenance=meta.get("provenance", {}),
        )

    def __repr__(self) -> str:
        return (
            f"Connectome({self.spec.name}, sex={self.spec.sex}, "
            f"{self.n_neurons:,} neurons, {self.n_edges:,} edges, "
            f"{self.n_synapses:,} synapses)"
        )


def file_digest(path: Path | str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a downloaded file, recorded in provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def config_digest(config: dict) -> str:
    """Stable hash of a build configuration, so results trace to their inputs."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
