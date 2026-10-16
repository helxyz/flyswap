"""Matching cell types across connectomes.

A graft needs to know that "COURT_B in the male" and "COURT_B in the female"
are the same type. In the synthetic data that is trivially true. On the real
datasets it is the hardest part of the pipeline, and it is not something this
project should try to solve from scratch: the published releases already
carry cross-dataset type annotations, and the MaleCNS paper reports the
dimorphism labels (isomorphic, dimorphic, male-specific, female-specific)
for thousands of types.

So this module *consumes* those published labels rather than inferring
matches. What it adds is the bookkeeping the experiments need: which types
exist in which datasets, how many neurons each has, and — importantly — an
honest coverage report, because the types that fail to match are a limitation
to state, not a failure to hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from flyswap.graph.build import SignedGraph

#: Dimorphism categories as reported by the MaleCNS paper.
DIMORPHISM_LABELS = ("isomorphic", "dimorphic", "male_specific", "female_specific")


@dataclass
class TypeMatch:
    """One cell type and its presence across datasets."""

    cell_type: str
    counts: dict[str, int]  # dataset -> neuron count
    label: str = "unknown"

    @property
    def datasets(self) -> list[str]:
        return sorted(d for d, n in self.counts.items() if n > 0)

    def present_in(self, *names: str) -> bool:
        return all(self.counts.get(n, 0) > 0 for n in names)

    def graftable_between(self, host: str, donor: str) -> bool:
        """Both sides need neurons, or there is nothing to graft."""
        return self.present_in(host, donor)

    def describe(self) -> dict:
        return {
            "cell_type": self.cell_type,
            "counts": dict(self.counts),
            "label": self.label,
            "datasets": self.datasets,
        }


@dataclass
class TypeSpaceReport:
    matches: dict[str, TypeMatch]
    datasets: list[str]
    coverage: dict = field(default_factory=dict)

    def shared(self, *names: str) -> list[str]:
        names = names or tuple(self.datasets)
        return sorted(t for t, m in self.matches.items() if m.present_in(*names))

    def with_label(self, label: str) -> list[str]:
        return sorted(t for t, m in self.matches.items() if m.label == label)

    def graftable(self, host: str, donor: str) -> list[str]:
        return sorted(
            t for t, m in self.matches.items() if m.graftable_between(host, donor)
        )

    def describe(self) -> dict:
        return {
            "datasets": self.datasets,
            "n_types": len(self.matches),
            "coverage": self.coverage,
            "by_label": {
                label: len(self.with_label(label))
                for label in (*DIMORPHISM_LABELS, "unknown")
            },
        }


def build_type_space(
    graphs: dict[str, SignedGraph],
    labels: dict[str, str] | None = None,
) -> TypeSpaceReport:
    """Index every cell type across the supplied graphs.

    `labels` maps cell type to a dimorphism category, as published. When it is
    absent, labels are inferred from presence: a type found in only one sex is
    marked sex-specific, everything shared is left "unknown" rather than
    guessed at, because distinguishing isomorphic from dimorphic needs the
    published annotation, not a presence check.
    """
    datasets = sorted(graphs)
    all_types: set[str] = set()
    counts: dict[str, dict[str, int]] = {}

    for name, g in graphs.items():
        vc = g.neurons["cell_type"].value_counts()
        for t, n in vc.items():
            t = str(t)
            all_types.add(t)
            counts.setdefault(t, {})[name] = int(n)

    sexes = {name: g.sex for name, g in graphs.items()}
    matches: dict[str, TypeMatch] = {}
    for t in sorted(all_types):
        per_dataset = {name: counts[t].get(name, 0) for name in datasets}
        if labels is not None:
            label = labels.get(t, "unknown")
        else:
            present_sexes = {
                sexes[name] for name, n in per_dataset.items() if n > 0
            }
            if present_sexes == {"male"}:
                label = "male_specific"
            elif present_sexes == {"female"}:
                label = "female_specific"
            else:
                label = "unknown"
        matches[t] = TypeMatch(cell_type=t, counts=per_dataset, label=label)

    n_all = sum(1 for m in matches.values() if m.present_in(*datasets))
    coverage = {
        "n_types_total": len(matches),
        "n_types_in_all_datasets": n_all,
        "fraction_in_all_datasets": round(n_all / max(len(matches), 1), 4),
        "per_dataset_type_counts": {
            name: sum(1 for m in matches.values() if m.counts.get(name, 0) > 0)
            for name in datasets
        },
        "labels_supplied": labels is not None,
    }

    return TypeSpaceReport(matches=matches, datasets=datasets, coverage=coverage)


def load_labels(path: Path | str, type_column: str = "cell_type") -> dict[str, str]:
    """Read published dimorphism labels from a CSV or Parquet table.

    Expected columns: a type name and a `dimorphism` column whose values fall
    in DIMORPHISM_LABELS. Unrecognised values become "unknown" rather than
    raising, so an unfamiliar vocabulary degrades gracefully.
    """
    path = Path(path)
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if type_column not in table.columns:
        raise ValueError(
            f"{path} has no {type_column!r} column; found {sorted(table.columns)[:20]}"
        )
    label_col = next(
        (c for c in ("dimorphism", "label", "category") if c in table.columns), None
    )
    if label_col is None:
        raise ValueError(f"{path} has no dimorphism/label/category column")

    normalise = {
        "isomorphic": "isomorphic",
        "dimorphic": "dimorphic",
        "male-specific": "male_specific",
        "male_specific": "male_specific",
        "female-specific": "female_specific",
        "female_specific": "female_specific",
    }
    out = {}
    for t, raw in zip(table[type_column], table[label_col], strict=True):
        out[str(t)] = normalise.get(str(raw).strip().lower(), "unknown")
    return out


def match_report(report: TypeSpaceReport, host: str, donor: str) -> pd.DataFrame:
    """A table of graftable types with their neuron counts, for inspection."""
    rows = []
    for t in report.graftable(host, donor):
        m = report.matches[t]
        rows.append(
            {
                "cell_type": t,
                "label": m.label,
                f"n_{host}": m.counts.get(host, 0),
                f"n_{donor}": m.counts.get(donor, 0),
                "count_ratio": round(
                    m.counts.get(host, 0) / max(m.counts.get(donor, 1), 1), 3
                ),
            }
        )
    return pd.DataFrame(rows)
