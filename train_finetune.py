"""OS-5: treinamento V1/V2 (protocolo de fine-tuning em dois estagios).

Consome o dataset consolidado por build_json.py (OS-4), formato
`{"_meta": {...}, "train": {...}, "test": {...}}` com janelas
`[window_size, 45]` e vetor de contexto por janela (Secao 6 do roadmap).

Nao modifica nenhum arquivo do repositorio original
(IC_Kalile_Intention_Prediction_HRC/) nem do hrc-data-collection/: a copia de
DLinear.py usada aqui vive em hrc-finetune/DLinear.py (OS-6).

Variantes:
    V0 - checkpoint original, sem ajuste (apenas avaliacao).
    V1 - fine-tuned nos dados novos, context_dim=0.
    V2 - V1 como inicializacao + context_proj ativo, fine-tuned com contexto
         (context_dim em {7, 10}).

Retrocompatibilidade (aceite da OS-5): com context_dim=0, Model_FinalIntention
desta copia produz a mesma saida numerica do checkpoint original para a
mesma entrada (nenhuma camada nova e instanciada); todo carregamento de
checkpoint usa strict=False.

Exemplo de uso:
    # V0: avalia o checkpoint original, sem treinar
    python train_finetune.py --variant V0 \\
        --dataset datasets/v1/dataset_dim0.json \\
        --init-checkpoint /path/to/original.pth \\
        --out-dir runs/V0

    # V1: fine-tune sem contexto, 3 sementes
    python train_finetune.py --variant V1 --context-dim 0 \\
        --dataset datasets/v1/dataset_dim0.json \\
        --init-checkpoint /path/to/original.pth \\
        --seeds 0 1 2 --out-dir runs/V1

    # V2: fine-tune com contexto, inicializando de cada seed de V1
    python train_finetune.py --variant V2 --context-dim 7 \\
        --dataset datasets/v1/dataset_dim7.json \\
        --init-checkpoint-per-seed runs/V1/seed0/checkpoint.pth \\
                                    runs/V1/seed1/checkpoint.pth \\
                                    runs/V1/seed2/checkpoint.pth \\
        --seeds 0 1 2 --out-dir runs/V2_dim7
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from DLinear import Model_FinalIntention
from predict import IntentionPredictor

INTENTION_LIST = {"no_action": 0, "get_connectors": 1, "get_screws": 2, "get_wheels": 3}
CONTEXT_DIMS = (0, 7, 10)
VARIANTS = ("V0", "V1", "V2")


class ModelArgs:
    """Namespace minimo consumido por Model_FinalIntention (seq_len, pred_len, ...)."""

    def __init__(self, seq_len=5, pred_len=5, class_num=4, individual=False, channels=45):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.class_num = class_num
        self.individual = individual
        self.channels = channels


class WindowDataset(Dataset):
    """Le janelas ja construidas pelo build_json.py (OS-4).

    Cada amostra retorna (pose, label, context). `context` tem shape
    (context_dim,); quando context_dim=0 e um tensor vazio, shape (0,).
    """

    def __init__(self, windows: List[Dict[str, Any]], context_dim: int):
        self.windows = windows
        self.context_dim = context_dim

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        pose = torch.tensor(window["pose"], dtype=torch.float32)
        label = torch.tensor(window["label"], dtype=torch.long)
        context = torch.tensor(window.get("context", []), dtype=torch.float32)
        if self.context_dim == 0:
            context = torch.zeros(0, dtype=torch.float32)
        return pose, label, context


def load_dataset_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_windows(dataset: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """Achata {label: {session_id: {windows: [...]}}} em uma lista de janelas."""
    windows: List[Dict[str, Any]] = []
    split_data = dataset.get(split, {})
    for _label, sessions in split_data.items():
        for _session_id, record in sessions.items():
            windows.extend(record.get("windows", []))
    return windows


def compute_class_weights(windows: List[Dict[str, Any]]) -> torch.Tensor:
    counts = [0] * len(INTENTION_LIST)
    for window in windows:
        counts[window["label"]] += 1
    weights = [1.0 / count if count > 0 else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(context_dim: int, model_args: ModelArgs) -> Model_FinalIntention:
    return Model_FinalIntention(model_args, context_dim=context_dim)


def load_checkpoint_into(model: nn.Module, checkpoint_path: Path) -> None:
    """Carrega um state_dict com strict=False (aceite obrigatorio da OS-5/OS-6)."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(
            f"checkpoint {checkpoint_path} has unexpected keys not present in the "
            f"model: {unexpected}"
        )


def freeze_backbone(model: Model_FinalIntention) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("context_proj") or name.startswith(
            "Intention_Predictor"
        )


def unfreeze_all(model: Model_FinalIntention) -> None:
    for param in model.parameters():
        param.requires_grad = True


def build_optimizer(
    model: Model_FinalIntention,
    lr: float,
    context_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """AdamW com taxa de aprendizado maior para context_proj quando presente.

    Secao 6 do roadmap: "Em V2, considerar taxa de aprendizado maior apenas
    para context_proj (camada nova) e menor para o restante".
    """
    if context_lr is not None and model.context_dim > 0:
        context_params = list(model.context_proj.parameters())
        context_param_ids = {id(p) for p in context_params}
        rest_params = [p for p in model.parameters() if id(p) not in context_param_ids]
        param_groups = [
            {"params": rest_params, "lr": lr},
            {"params": context_params, "lr": context_lr},
        ]
        return torch.optim.AdamW(param_groups, lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


@dataclass
class EvalMetrics:
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: List[List[int]]
    ece: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "per_class_accuracy": self.per_class_accuracy,
            "confusion_matrix": self.confusion_matrix,
            "ece": self.ece,
            "num_samples": self.num_samples,
        }


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error com bins de largura uniforme em [0, 1]."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(ece)


@torch.no_grad()
def evaluate(
    model: Model_FinalIntention,
    loader: DataLoader,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    class_names = list(INTENTION_LIST.keys())
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_confidences = []
    all_correct = []
    total = 0
    correct_total = 0

    # Avalia via IntentionPredictor.predict (restrict="ood") para incluir o
    # corte por entropia do preditor completo do paper, nao so a rede
    # DLinear crua (predict() exige batch_size=1, como o loader ja usa).
    predictor = IntentionPredictor(model=model)

    for pose, label, context in loader:
        pose = pose.to(device)
        label = label.to(device)
        ctx = context.to(device) if context.numel() > 0 else None
        _, batch_intention = predictor.predict(pose, restrict="ood", context=ctx)
        _, logits = model(pose, ctx)
        confidence = torch.softmax(logits, dim=1).max(dim=1).values
        pred = batch_intention

        for true_label, pred_label in zip(label.tolist(), pred.tolist()):
            confusion[true_label][pred_label] += 1

        correct = (pred == label).float()
        all_confidences.append(confidence.cpu().numpy())
        all_correct.append(correct.cpu().numpy())
        correct_total += correct.sum().item()
        total += label.shape[0]

    accuracy = correct_total / total if total > 0 else 0.0
    per_class_accuracy = {}
    for idx, name in enumerate(class_names):
        class_total = confusion[idx].sum()
        per_class_accuracy[name] = (
            float(confusion[idx][idx] / class_total) if class_total > 0 else 0.0
        )

    confidences = np.concatenate(all_confidences) if all_confidences else np.array([])
    correct_arr = np.concatenate(all_correct) if all_correct else np.array([])
    ece = compute_ece(confidences, correct_arr)

    return EvalMetrics(
        accuracy=accuracy,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion.tolist(),
        ece=ece,
        num_samples=total,
    )


def train_one_epoch(
    model: Model_FinalIntention,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    for pose, label, context in loader:
        pose = pose.to(device)
        label = label.to(device)
        ctx = context.to(device) if context.numel() > 0 else None

        optimizer.zero_grad()
        _, logits = model(pose, ctx)
        loss = class_criterion(logits, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1
    return running_loss / count if count > 0 else 0.0


def run_variant(
    variant: str,
    context_dim: int,
    train_windows: List[Dict[str, Any]],
    test_windows: List[Dict[str, Any]],
    init_checkpoint: Optional[Path],
    seed: int,
    epochs: int,
    lr: float,
    context_lr: Optional[float],
    batch_size: int,
    freeze_epochs: int,
    out_dir: Path,
    model_args: ModelArgs,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)

    model = build_model(context_dim, model_args).to(device)
    if init_checkpoint is not None:
        load_checkpoint_into(model, init_checkpoint)

    test_dataset = WindowDataset(test_windows, context_dim)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if variant == "V0":
        # V0 e apenas avaliacao do checkpoint original, sem nenhum treino.
        metrics = evaluate(model, test_loader, device)
        result = {
            "variant": variant,
            "context_dim": context_dim,
            "seed": seed,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "hyperparameters": {},
            "metrics": metrics.to_dict(),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result

    train_dataset = WindowDataset(train_windows, context_dim)
    class_weights = compute_class_weights(train_windows).to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    if variant == "V2" and context_dim > 0 and freeze_epochs > 0:
        # Fase 1: backbone congelado, so context_proj + Intention_Predictor treinam.
        freeze_backbone(model)
        optimizer = build_optimizer(model, lr=lr, context_lr=context_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(freeze_epochs, 1))
        for epoch in range(freeze_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, class_criterion, device)
            scheduler.step()
            log_lines.append(f"[phase1 frozen] epoch {epoch + 1}/{freeze_epochs} loss={loss:.6f}")
        unfreeze_all(model)
        remaining_epochs = max(epochs - freeze_epochs, 0)
    else:
        remaining_epochs = epochs

    optimizer = build_optimizer(model, lr=lr, context_lr=context_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1))
    for epoch in range(remaining_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, class_criterion, device)
        scheduler.step()
        log_lines.append(f"epoch {epoch + 1}/{remaining_epochs} loss={loss:.6f}")

    metrics = evaluate(model, test_loader, device)

    checkpoint_path = seed_dir / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    (seed_dir / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    result = {
        "variant": variant,
        "context_dim": context_dim,
        "seed": seed,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "checkpoint": str(checkpoint_path),
        "hyperparameters": {
            "optimizer": "AdamW",
            "lr": lr,
            "context_lr": context_lr,
            "scheduler": "CosineAnnealingLR",
            "epochs": epochs,
            "freeze_epochs": freeze_epochs,
            "batch_size": batch_size,
        },
        "metrics": metrics.to_dict(),
    }
    with open(seed_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OS-5: treinamento V0/V1/V2 com contexto parametrizavel.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="JSON consolidado (build_json.py)")
    parser.add_argument("--context-dim", type=int, choices=CONTEXT_DIMS, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="checkpoint unico (V0, V1)")
    parser.add_argument(
        "--init-checkpoint-per-seed",
        type=Path,
        nargs="*",
        default=None,
        help="checkpoints de V1 pareados por indice de seed, para V2 (Secao 6: protocolo de seed pareado)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=4, help="epocas de fase 1 (backbone congelado) para V2")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-lr", type=float, default=None, help="lr especifico para context_proj (V2)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--class-num", type=int, default=4)
    parser.add_argument("--channels", type=int, default=45)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.variant in ("V1", "V2") and args.context_dim == 0 and args.variant == "V2":
        raise ValueError("V2 requires --context-dim in {7, 10}; use V1 for context_dim=0")

    if args.variant == "V2" and not args.init_checkpoint_per_seed:
        raise ValueError(
            "V2 requires --init-checkpoint-per-seed with one checkpoint per --seeds entry "
            "(Secao 6: V2 inicializa do checkpoint de V1, pareado por seed)"
        )
    if args.variant == "V2" and len(args.init_checkpoint_per_seed) != len(args.seeds):
        raise ValueError("--init-checkpoint-per-seed must have exactly one entry per --seeds")

    dataset = load_dataset_json(args.dataset)
    meta = dataset.get("_meta", {})
    dataset_context_dim = meta.get("context_dim")
    if dataset_context_dim is not None and dataset_context_dim != args.context_dim:
        raise ValueError(
            f"--context-dim {args.context_dim} does not match dataset context_dim "
            f"{dataset_context_dim} recorded in {args.dataset}"
        )

    train_windows = flatten_windows(dataset, "train")
    test_windows = flatten_windows(dataset, "test")
    if args.variant != "V0" and not train_windows:
        raise ValueError(f"no training windows found in {args.dataset}")
    if not test_windows:
        raise ValueError(f"no test windows found in {args.dataset}")

    model_args = ModelArgs(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        class_num=args.class_num,
        channels=args.channels,
    )
    device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "variant": args.variant,
        "context_dim": args.context_dim,
        "dataset": str(args.dataset),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "lr": args.lr,
        "context_lr": args.context_lr,
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    results = []
    if args.variant == "V0":
        result = run_variant(
            variant="V0",
            context_dim=args.context_dim,
            train_windows=train_windows,
            test_windows=test_windows,
            init_checkpoint=args.init_checkpoint,
            seed=args.seeds[0],
            epochs=0,
            lr=args.lr,
            context_lr=args.context_lr,
            batch_size=args.batch_size,
            freeze_epochs=0,
            out_dir=args.out_dir,
            model_args=model_args,
            device=device,
        )
        results.append(result)
    else:
        for idx, seed in enumerate(args.seeds):
            if args.variant == "V1":
                init_ckpt = args.init_checkpoint
            else:
                init_ckpt = args.init_checkpoint_per_seed[idx]
            result = run_variant(
                variant=args.variant,
                context_dim=args.context_dim,
                train_windows=train_windows,
                test_windows=test_windows,
                init_checkpoint=init_ckpt,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                context_lr=args.context_lr,
                batch_size=args.batch_size,
                freeze_epochs=args.freeze_epochs,
                out_dir=args.out_dir,
                model_args=model_args,
                device=device,
            )
            results.append(result)

    if len(results) > 1:
        accuracies = [r["metrics"]["accuracy"] for r in results]
        eces = [r["metrics"]["ece"] for r in results]
        summary = {
            "variant": args.variant,
            "context_dim": args.context_dim,
            "num_seeds": len(results),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "ece_mean": float(np.mean(eces)),
            "ece_std": float(np.std(eces)),
        }
        with open(args.out_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(results[0]["metrics"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
