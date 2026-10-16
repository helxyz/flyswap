"""Command line entry point.

Every result in the project is reproducible as one command, and every command
writes a JSON record carrying the configuration that produced it. That record
is what makes a number citable later: it says which datasets, which sign rule,
which gain, which seeds.

    flyswap calibrate   --data synthetic
    flyswap dimorphism  --data synthetic --out results/
    flyswap graft       --data synthetic --out results/
    flyswap attribute   --data synthetic --out results/
    flyswap bench       --out results/
    flyswap all         --data synthetic --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from flyswap import __version__


def _load_graphs(source: str, data_dir: str | None, min_syn: int):
    """Synthetic fixtures, or real connectomes from a directory."""
    from flyswap.graph import build_graph

    if source == "synthetic":
        from flyswap.data.synthetic import make_pair

        pair = make_pair()
        return {name: build_graph(c) for name, c in pair.as_dict().items()}

    if data_dir is None:
        raise SystemExit("--data-dir is required unless --data synthetic")

    from flyswap.data.loaders import load_banc, load_fafb, load_malecns

    root = Path(data_dir)
    graphs = {}
    loaders = {
        "male": (load_malecns, root / "malecns"),
        "female_a": (load_fafb, root / "fafb"),
        "female_b": (load_banc, root / "banc"),
    }
    for name, (fn, path) in loaders.items():
        if path.exists():
            graphs[name] = build_graph(fn(path, min_syn=min_syn))
        else:
            print(f"  skipping {name}: {path} not found", file=sys.stderr)
    if not graphs:
        raise SystemExit(f"no datasets found under {root}")
    return graphs


def _conditions(source: str):
    """Stimulus/readout pairs to test.

    For the real connectome these are the circuits with published behavioural
    meaning. Keeping them in one place means an experiment cannot quietly
    acquire a new readout after seeing results.
    """
    from flyswap.experiments import StimulusCondition

    if source == "synthetic":
        return [
            StimulusCondition(
                "sensory_drive", "SENS", ("SONG", "MOTOR", "PREMOT", "COURT_B", "RELAY")
            )
        ]
    return [
        StimulusCondition("sugar", "sugar_GRN", ("MN9",)),
        StimulusCondition("bitter", "bitter_GRN", ("MN9",)),
        StimulusCondition("looming", "LC4", ("GiantFiber", "TTMn")),
        StimulusCondition("courtship", "P1", ("pIP10", "vpoDN")),
    ]


def _write(out_dir: Path | None, name: str, payload: dict) -> None:
    payload = {
        "flyswap_version": __version__,
        "generated_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    if out_dir is None:
        print(json.dumps(payload, indent=2, default=str))
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {path}")


def cmd_calibrate(args) -> dict:
    from flyswap.sim import calibrate_gain

    graphs = _load_graphs(args.data, args.data_dir, args.min_syn)
    stim, down = (
        ("SENS", "MOTOR") if args.data == "synthetic" else (args.stimulus, args.readout)
    )
    results = {}
    for name, g in graphs.items():
        res = calibrate_gain(g, stim, down)
        results[name] = res.describe()
        print(f"  {name}: gain = {res.gain}")

    chosen = {r["chosen_gain"] for r in results.values() if r["chosen_gain"]}
    agreed = len(chosen) == 1
    print(f"  agreement across animals: {agreed} ({sorted(chosen)})")
    return {
        "command": "calibrate",
        "per_dataset": results,
        "agreed_gain": chosen.pop() if agreed else None,
        "agreement": agreed,
    }


def cmd_dimorphism(args) -> dict:
    from flyswap.experiments import run_dimorphism_experiment
    from flyswap.sim import RateParams

    graphs = _load_graphs(args.data, args.data_dir, args.min_syn)
    males = {k: v for k, v in graphs.items() if v.sex == "male"}
    females = {k: v for k, v in graphs.items() if v.sex == "female"}
    if not males:
        raise SystemExit("no male dataset loaded")
    if len(females) < 2:
        raise SystemExit(
            f"need two female datasets for the within-sex baseline; got {sorted(females)}"
        )

    male = next(iter(males.values()))
    report = run_dimorphism_experiment(
        male,
        females,
        _conditions(args.data),
        RateParams(gain=args.gain),
        seeds=tuple(range(args.seeds)),
    )
    print(f"  {'readout':>10} {'score':>8} {'ratio':>8}  dimorphic")
    for s in report.scores:
        print(
            f"  {s.readout_type:>10} {s.score:8.3f} {s.effect_ratio:8.2f}  {s.significant}"
        )
    return {"command": "dimorphism", **report.describe()}


def cmd_graft(args) -> dict:
    from flyswap.interventions import TypeSpace, graft_type, null_graft
    from flyswap.sim import RateParams, RateSimulator

    graphs = _load_graphs(args.data, args.data_dir, args.min_syn)
    params = RateParams(gain=args.gain)
    space = TypeSpace(graphs)
    condition = _conditions(args.data)[0]
    readout_type = condition.readout_types[0]

    def readout(g):
        sim = RateSimulator(g, params)
        steady = sim.steady_state(
            sim.stimulus_for_type(condition.stimulus_type, condition.amplitude)
        )
        return float(sim.readout(steady, readout_type))

    host, donor = "male", "female_a"
    base_host, base_donor = readout(graphs[host]), readout(graphs[donor])
    span = base_host - base_donor

    effects = []
    for t in space.shared_types(host, donor):
        grafted, spec = graft_type(graphs[host], graphs[donor], t, space, host, donor)
        value = readout(grafted)
        effects.append(
            {
                "cell_type": t,
                "readout_after_graft": round(value, 4),
                "effect": round(base_host - value, 4),
                "fraction_toward_donor": round(
                    (base_host - value) / span if abs(span) > 1e-9 else 0.0, 4
                ),
                "entries_changed": spec.notes["entries_changed"],
            }
        )
    effects.sort(key=lambda e: -abs(e["effect"]))

    nulls = []
    if "female_b" in graphs:
        nb = readout(graphs["female_a"])
        for e in effects[: args.top]:
            grafted, _ = null_graft(
                graphs["female_a"],
                graphs["female_b"],
                e["cell_type"],
                space,
                "female_a",
                "female_b",
            )
            nulls.append(
                {
                    "cell_type": e["cell_type"],
                    "null_effect": round(nb - readout(grafted), 4),
                    "real_effect": e["effect"],
                }
            )

    print(f"  top grafts ({host} <- {donor}), readout {readout_type}:")
    for e in effects[: args.top]:
        print(
            f"    {e['cell_type']:>12}  effect {e['effect']:9.3f}  "
            f"{e['fraction_toward_donor'] * 100:6.1f}% toward donor"
        )
    return {
        "command": "graft",
        "host": host,
        "donor": donor,
        "readout_type": readout_type,
        "baseline_host": round(base_host, 4),
        "baseline_donor": round(base_donor, 4),
        "effects": effects,
        "null_grafts": nulls,
        "params": params.describe(),
    }


def cmd_attribute(args) -> dict:
    from flyswap.attribution import compare_to_grafts, integrated_gradients
    from flyswap.interventions import TypeSpace, graft_type
    from flyswap.sim import RateParams, RateSimulator

    graphs = _load_graphs(args.data, args.data_dir, args.min_syn)
    params = RateParams(gain=args.gain)
    space = TypeSpace(graphs)
    condition = _conditions(args.data)[0]
    readout_type = condition.readout_types[0]

    sim = RateSimulator(graphs["male"], params)
    attribution = integrated_gradients(
        sim, condition.stimulus_type, readout_type, steps=args.ig_steps
    )

    def readout(g):
        s = RateSimulator(g, params)
        return float(
            s.readout(
                s.steady_state(
                    s.stimulus_for_type(condition.stimulus_type, condition.amplitude)
                ),
                readout_type,
            )
        )

    base = readout(graphs["male"])
    effects = {}
    for t in space.shared_types("male", "female_a"):
        grafted, _ = graft_type(
            graphs["male"], graphs["female_a"], t, space, "male", "female_a"
        )
        effects[t] = base - readout(grafted)

    agreement = compare_to_grafts(attribution, effects, top_fraction=args.top_fraction)
    print(f"  gradient top 5: {attribution.top(5)}")
    print(f"  graft top 5:    {agreement.top_by_graft[:5]}")
    print(f"  spearman rho:   {agreement.spearman_rho:.4f}")
    return {
        "command": "attribute",
        "attribution": {
            "method": attribution.method,
            "readout_type": attribution.readout_type,
            "ranking": [
                {"cell_type": t, "score": round(v, 6)}
                for t, v in attribution.ranking()
            ],
            "config": attribution.config,
        },
        "graft_effects": {k: round(v, 4) for k, v in effects.items()},
        "agreement": agreement.describe(),
    }


def cmd_bench(args) -> dict:
    from flyswap.bench import (
        bench_batch,
        bench_gradient_vs_grafts,
        bench_scaling,
        bench_single,
        environment,
    )

    graphs = _load_graphs(args.data, args.data_dir, args.min_syn)
    graph = graphs["male"]
    stim, readout = (
        ("SENS", "MOTOR") if args.data == "synthetic" else (args.stimulus, args.readout)
    )

    single = bench_single(graph)
    batch = bench_batch(graph, batch_sizes=tuple(args.batch_sizes))
    grad = bench_gradient_vs_grafts(graph, stim, readout)
    scaling = bench_scaling(sizes=tuple(args.scaling_sizes)) if args.scaling else []

    print(f"  single: {single.neuron_steps_per_s:,.0f} neuron-steps/s")
    for b in batch:
        print(f"  {b.label}: {b.neuron_steps_per_s:,.0f} neuron-steps/s")
    for s in scaling:
        print(f"  {s.label}: {s.neuron_steps_per_s:,.0f} neuron-steps/s")
    if grad["gradient_is_faster"]:
        print(
            f"  gradient vs brute force: {grad['speedup']}x faster "
            f"({grad['n_types']} types)"
        )
    else:
        print(
            f"  gradient vs brute force: {grad['speedup']}x — slower here; "
            f"pays off above ~{grad['break_even_n_types']:.0f} cell types "
            f"(this graph has {grad['n_types']})"
        )

    return {
        "command": "bench",
        "environment": environment(),
        "single": single.describe(),
        "batch": [b.describe() for b in batch],
        "scaling": [s.describe() for s in scaling],
        "gradient_vs_grafts": grad,
    }


def cmd_all(args) -> dict:
    out = {}
    for name, fn in (
        ("calibrate", cmd_calibrate),
        ("dimorphism", cmd_dimorphism),
        ("graft", cmd_graft),
        ("attribute", cmd_attribute),
        ("bench", cmd_bench),
    ):
        print(f"\n[{name}]")
        result = fn(args)
        _write(args.out, name, result)
        out[name] = result
    return {"command": "all", "ran": sorted(out)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flyswap",
        description="Counterfactual cross-sex simulation of the Drosophila connectome.",
    )
    parser.add_argument("--version", action="version", version=f"flyswap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument(
            "--data",
            default="synthetic",
            choices=["synthetic", "real"],
            help="synthetic fixtures (default) or downloaded connectomes",
        )
        p.add_argument("--data-dir", default=None, help="root of downloaded datasets")
        p.add_argument("--min-syn", type=int, default=1)
        p.add_argument("--out", type=Path, default=None, help="directory for JSON records")
        p.add_argument("--gain", type=float, default=2.2)
        return p

    c = common(sub.add_parser("calibrate", help="pick the global gain by rule"))
    c.add_argument("--stimulus", default="sugar_GRN")
    c.add_argument("--readout", default="MN9")
    c.set_defaults(func=cmd_calibrate)

    d = common(sub.add_parser("dimorphism", help="E1: variability-controlled comparison"))
    d.add_argument("--seeds", type=int, default=10)
    d.set_defaults(func=cmd_dimorphism)

    g = common(sub.add_parser("graft", help="E2: cross-sex cell-type grafts"))
    g.add_argument("--top", type=int, default=10)
    g.set_defaults(func=cmd_graft)

    a = common(sub.add_parser("attribute", help="E3: gradients, scored against grafts"))
    a.add_argument("--ig-steps", type=int, default=32)
    a.add_argument("--top-fraction", type=float, default=0.25)
    a.set_defaults(func=cmd_attribute)

    b = common(sub.add_parser("bench", help="throughput and scaling"))
    b.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    b.add_argument("--scaling", action="store_true")
    b.add_argument("--scaling-sizes", type=int, nargs="+", default=[500, 2000, 8000])
    b.add_argument("--stimulus", default="sugar_GRN")
    b.add_argument("--readout", default="MN9")
    b.set_defaults(func=cmd_bench)

    x = common(sub.add_parser("all", help="run every experiment in order"))
    x.add_argument("--seeds", type=int, default=10)
    x.add_argument("--top", type=int, default=10)
    x.add_argument("--ig-steps", type=int, default=32)
    x.add_argument("--top-fraction", type=float, default=0.25)
    x.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    x.add_argument("--scaling", action="store_true")
    x.add_argument("--scaling-sizes", type=int, nargs="+", default=[500, 2000])
    x.add_argument("--stimulus", default="sugar_GRN")
    x.add_argument("--readout", default="MN9")
    x.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.random.seed(0)
    result = args.func(args)
    if args.command != "all":
        _write(args.out, args.command, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
