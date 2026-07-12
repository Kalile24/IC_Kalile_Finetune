"""Agrega runs/*/seed*/metrics.json (e runs/V0/eval_metrics.json) em:

  - results/ablation_table.md      tabela V0/V1/V2_dim7/V2_dim10 (Secao 6 do
                                    roadmap: media +/- desvio-padrao por variante)
  - results/confusion_matrices.png matrizes de confusao normalizadas, uma por
                                    variante, somadas sobre as sementes

Nao faz parte do pipeline de treino (train_finetune.py); e uma ferramenta de
relatorio, pensada para rodar depois que os diretorios runs/V0, runs/V1,
runs/V2_dim7 (e opcionalmente runs/V2_dim10) existirem.

Uso:
    python reports/scripts/aggregate_results.py \\
        --runs-root runs \\
        --out-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn import metrics as sk_metrics
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

INTENTION_LIST = {"no_action": 0, "get_connectors": 1, "get_screws": 2, "get_wheels": 3}
CLASS_NAMES = [name for name, _ in sorted(INTENTION_LIST.items(), key=lambda kv: kv[1])]
VARIANT_DIRS = ["V0", "V1", "V2_dim7", "V2_dim10"]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_v0(runs_root: Path) -> Optional[Dict[str, Any]]:
    return _load_json(runs_root / "V0" / "eval_metrics.json")


def load_variant_seeds(runs_root: Path, variant_dir: str) -> List[Dict[str, Any]]:
    base = runs_root / variant_dir
    if not base.is_dir():
        return []
    results = []
    for seed_dir in sorted(base.glob("seed*")):
        metrics_path = seed_dir / "metrics.json"
        result = _load_json(metrics_path)
        if result is not None:
            results.append(result)
    return results


def summarize(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    accuracies = [r["metrics"]["accuracy"] for r in results]
    eces = [r["metrics"]["ece"] for r in results]
    per_class = {name: [] for name in CLASS_NAMES}
    for r in results:
        for name in CLASS_NAMES:
            per_class[name].append(r["metrics"]["per_class_accuracy"].get(name, 0.0))
    return {
        "num_seeds": len(results),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "ece_mean": float(np.mean(eces)),
        "ece_std": float(np.std(eces)),
        "per_class_mean": {name: float(np.mean(v)) for name, v in per_class.items()},
        "per_class_std": {name: float(np.std(v)) for name, v in per_class.items()},
    }


def render_ablation_table(
    v0: Optional[Dict[str, Any]],
    variant_summaries: Dict[str, Optional[Dict[str, Any]]],
) -> str:
    lines = [
        "# Tabela ablativa V0/V1/V2",
        "",
        "| Variante | Sementes | Top-1 (media +/- dp) | ECE (media +/- dp) | "
        + " | ".join(f"{name} (top-1)" for name in CLASS_NAMES)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(CLASS_NAMES),
    ]

    if v0 is not None:
        acc = v0["metrics"]["accuracy"]
        ece = v0["metrics"]["ece"]
        per_class = v0["metrics"]["per_class_accuracy"]
        row = (
            f"| V0 (baseline, sem ajuste) | 1 | {acc:.3f} | {ece:.3f} | "
            + " | ".join(f"{per_class.get(name, 0.0):.3f}" for name in CLASS_NAMES)
            + " |"
        )
        lines.append(row)
    else:
        lines.append("| V0 (baseline, sem ajuste) | - | n/d | n/d | " + " | ".join(["n/d"] * len(CLASS_NAMES)) + " |")

    labels = {
        "V1": "V1 (fine-tune, context_dim=0)",
        "V2_dim7": "V2 (contexto 7D)",
        "V2_dim10": "V2 (contexto 10D, ablacao)",
    }
    for key, label in labels.items():
        summary = variant_summaries.get(key)
        if summary is None:
            lines.append(f"| {label} | - | n/d | n/d | " + " | ".join(["n/d"] * len(CLASS_NAMES)) + " |")
            continue
        acc_str = f'{summary["accuracy_mean"]:.3f} +/- {summary["accuracy_std"]:.3f}'
        ece_str = f'{summary["ece_mean"]:.3f} +/- {summary["ece_std"]:.3f}'
        per_class_str = " | ".join(
            f'{summary["per_class_mean"][name]:.3f}' for name in CLASS_NAMES
        )
        lines.append(
            f"| {label} | {summary['num_seeds']} | {acc_str} | {ece_str} | {per_class_str} |"
        )

    if variant_summaries.get("V1") is not None and variant_summaries.get("V2_dim7") is not None:
        delta = (
            variant_summaries["V2_dim7"]["accuracy_mean"]
            - variant_summaries["V1"]["accuracy_mean"]
        )
        lines += ["", f"**Delta V2(7D) - V1 (claim central do artigo): {delta:+.3f}**"]

    return "\n".join(lines) + "\n"


def _confusion_from_counts(confusion: List[List[int]]) -> np.ndarray:
    matrix = np.array(confusion, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def plot_confusion_matrices(
    v0: Optional[Dict[str, Any]],
    variant_results: Dict[str, List[Dict[str, Any]]],
    out_path: Path,
) -> Optional[Path]:
    if not _HAS_PLOTTING:
        print("matplotlib/sklearn indisponiveis; pulando geracao de figura.")
        return None

    panels = []
    if v0 is not None:
        panels.append(("V0", _confusion_from_counts(v0["metrics"]["confusion_matrix"])))
    for key, label in [("V1", "V1"), ("V2_dim7", "V2 (7D)"), ("V2_dim10", "V2 (10D)")]:
        results = variant_results.get(key, [])
        if not results:
            continue
        summed = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
        for r in results:
            summed += np.array(r["metrics"]["confusion_matrix"], dtype=np.float64)
        panels.append((label, _confusion_from_counts(summed.tolist())))

    if not panels:
        print("nenhuma metrica encontrada; nada para plotar.")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, matrix) in zip(axes, panels):
        display = sk_metrics.ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=CLASS_NAMES
        )
        display.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(label)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Agrega runs/ em tabela ablativa e matrizes de confusao.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    v0 = load_v0(args.runs_root)
    variant_results = {key: load_variant_seeds(args.runs_root, key) for key in ("V1", "V2_dim7", "V2_dim10")}
    variant_summaries = {key: summarize(results) for key, results in variant_results.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = render_ablation_table(v0, variant_summaries)
    table_path = args.out_dir / "ablation_table.md"
    table_path.write_text(table, encoding="utf-8")
    print(f"tabela: {table_path}")

    fig_path = plot_confusion_matrices(v0, variant_results, args.out_dir / "confusion_matrices.png")
    if fig_path:
        print(f"figura: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
