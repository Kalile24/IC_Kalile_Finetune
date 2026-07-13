"""Testes de aceite da OS-5 (train_finetune.py).

Criterios do roadmap (Secao 8, linha da OS-5):
    "Aceite: retrocompatibilidade com context_dim=0 e reproducao da
    avaliacao de V0."

1. test_context_dim_zero_matches_original_model: com context_dim=0, a copia
   local de Model_FinalIntention (hrc-finetune/DLinear.py) produz saida
   numericamente identica ao Model_FinalIntention original do repositorio
   fonte, para o mesmo checkpoint e a mesma entrada.
2. test_v0_evaluation_is_reproducible: rodar train_finetune.py --variant V0
   duas vezes sobre o mesmo checkpoint/dataset produz metricas identicas
   (V0 nao treina, so avalia) - "reproducao da avaliacao de V0".
3. Testes de suporte: strict=False ao carregar checkpoint original em
   context_dim>0 (zero-init preserva a saida antes do fine-tuning) e
   validacoes basicas de CLI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

FINETUNE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FINETUNE_DIR))

import train_finetune as tf  # noqa: E402
from DLinear import Model_FinalIntention as LocalModel  # noqa: E402

ORIGINAL_TRAJ_INTENTION = Path(
    "/home/marcos-kalile/IC_Kalile_Intention_Prediction_HRC/traj_intention"
)
ORIGINAL_CHECKPOINT = (
    ORIGINAL_TRAJ_INTENTION
    / "checkpoints"
    / "seq5_pred5_epoch40_whole_pkl_final_intention_nomaskTrue.pth"
)

pytestmark = pytest.mark.skipif(
    not ORIGINAL_CHECKPOINT.exists(),
    reason="original checkpoint not available in this environment",
)


def _load_original_model_class():
    """Importa Model_FinalIntention do repositorio fonte, sem modifica-lo."""
    sys.path.insert(0, str(ORIGINAL_TRAJ_INTENTION))
    import DLinear as original_dlinear  # noqa: E402

    return original_dlinear.Model_FinalIntention


def test_context_dim_zero_matches_original_model():
    """Aceite principal: context_dim=0 == comportamento original bit a bit."""
    OriginalModel = _load_original_model_class()

    model_args = tf.ModelArgs(seq_len=5, pred_len=5, class_num=4, channels=45)

    original_model = OriginalModel(model_args)
    local_model = LocalModel(model_args, context_dim=0)

    state_dict = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu")
    original_model.load_state_dict(state_dict)
    missing, unexpected = local_model.load_state_dict(state_dict, strict=False)
    assert unexpected == [], f"local model has unexpected extra keys: {unexpected}"
    assert missing == [], "context_dim=0 must not introduce missing params"

    original_model.eval()
    local_model.eval()

    torch.manual_seed(0)
    x = torch.randn(4, 5, 45)

    with torch.no_grad():
        orig_traj, orig_intention = original_model(x)
        local_traj, local_intention = local_model(x, context=None)

    assert torch.allclose(orig_traj, local_traj, atol=1e-7)
    assert torch.allclose(orig_intention, local_intention, atol=1e-7)


def test_context_dim_nonzero_zero_init_matches_original_before_training():
    """context_proj zero-inicializado: V2 recem-carregado == V1/original numericamente.

    Roadmap Secao 3 (planejamento_finetune_os5-7.md): "Teste de aceite
    recomendado antes de treinar de fato: rodar o mesmo batch em model_v1 e
    em model_v2 recem-carregado (qualquer context) e conferir que
    intention_output bate exatamente."
    """
    model_args = tf.ModelArgs(seq_len=5, pred_len=5, class_num=4, channels=45)
    state_dict = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu")

    model_v1 = LocalModel(model_args, context_dim=0)
    model_v1.load_state_dict(state_dict, strict=False)
    model_v1.eval()

    for context_dim in (7, 10):
        model_v2 = LocalModel(model_args, context_dim=context_dim)
        missing, unexpected = model_v2.load_state_dict(state_dict, strict=False)
        assert unexpected == []
        assert set(missing) == {"context_proj.weight", "context_proj.bias"}
        model_v2.eval()

        torch.manual_seed(1)
        x = torch.randn(3, 5, 45)
        context = torch.randn(3, context_dim)

        with torch.no_grad():
            _, intention_v1 = model_v1(x)
            _, intention_v2 = model_v2(x, context=context)

        assert torch.allclose(intention_v1, intention_v2, atol=1e-7), (
            f"context_dim={context_dim}: zero-init context_proj must not perturb "
            "the inherited checkpoint's output"
        )


def _write_synthetic_dataset(path: Path, context_dim: int, num_train=40, num_test=20) -> None:
    """Gera um dataset sintetico no schema real do build_json.py (OS-4).

    Usado apenas para exercitar train_finetune.py de ponta a ponta neste
    ambiente, onde ainda nao existe um dataset consolidado real (aguardando
    coleta de sessoes). O schema replica exatamente o formato produzido por
    hrc-data-collection/src/datacol/build_json.py.
    """
    rng = np.random.default_rng(0)

    def make_windows(n, session_id):
        windows = []
        for i in range(n):
            label_name = list(tf.INTENTION_LIST.keys())[i % len(tf.INTENTION_LIST)]
            label_id = tf.INTENTION_LIST[label_name]
            pose = rng.normal(size=(5, 45)).tolist()
            if context_dim == 0:
                context = []
            else:
                context = rng.uniform(0, 1, size=context_dim).tolist()
            windows.append(
                {
                    "session_id": session_id,
                    "frame_idx": i,
                    "end_frame_idx": i + 4,
                    "intention": label_name,
                    "label": label_id,
                    "pose": pose,
                    "context": context,
                }
            )
        return windows

    def split_block(windows):
        block = {label: {} for label in tf.INTENTION_LIST}
        for window in windows:
            label_name = window["intention"]
            block[label_name].setdefault(
                window["session_id"], {"start": [1], "end": [len(windows) + 1], "windows": []}
            )
            block[label_name][window["session_id"]]["windows"].append(window)
        return block

    train_windows = make_windows(num_train, "S01_train")
    test_windows = make_windows(num_test, "S02_test")

    dataset = {
        "_meta": {
            "format_version": 1,
            "window_size": 5,
            "joints": 15,
            "coordinates": 3,
            "channels": 45,
            "context_dim": context_dim,
            "plan_policy": "proxy_graph",
            "intention_list": dict(tf.INTENTION_LIST),
            "splits": {"train": ["S01_train"], "test": ["S02_test"]},
        },
        "train": split_block(train_windows),
        "test": split_block(test_windows),
    }
    path.write_text(json.dumps(dataset), encoding="utf-8")


def test_v0_evaluation_is_reproducible(tmp_path):
    """Aceite: reproducao da avaliacao de V0.

    V0 nao treina (so avalia o checkpoint original) - rodar o comando duas
    vezes sobre o mesmo checkpoint/dataset deve produzir metricas
    identicas (mesma acuracia, mesma matriz de confusao, mesmo ECE).
    """
    dataset_path = tmp_path / "dataset_dim0.json"
    _write_synthetic_dataset(dataset_path, context_dim=0)

    out_dir_1 = tmp_path / "runs_v0_a"
    out_dir_2 = tmp_path / "runs_v0_b"

    for out_dir in (out_dir_1, out_dir_2):
        tf.main(
            [
                "--variant", "V0",
                "--dataset", str(dataset_path),
                "--context-dim", "0",
                "--init-checkpoint", str(ORIGINAL_CHECKPOINT),
                "--seeds", "0",
                "--out-dir", str(out_dir),
            ]
        )

    metrics_1 = json.loads((out_dir_1 / "eval_metrics.json").read_text())
    metrics_2 = json.loads((out_dir_2 / "eval_metrics.json").read_text())

    assert metrics_1["metrics"] == metrics_2["metrics"]
    assert metrics_1["metrics"]["num_samples"] == 20


def test_evaluate_applies_entropy_cutoff():
    """evaluate() deve avaliar o preditor completo (IntentionPredictor,
    restrict="ood"), nao so a rede DLinear crua - caso contrario o corte por
    entropia do artigo (predict.py) nunca e exercitado nas metricas de
    V0/V1/V2."""
    from torch.utils.data import DataLoader

    model_args = tf.ModelArgs(seq_len=5, pred_len=5, class_num=4, channels=45)
    model = LocalModel(model_args, context_dim=0)
    state_dict = torch.load(ORIGINAL_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    windows = []
    torch.manual_seed(5)
    for i in range(8):
        windows.append(
            {
                "pose": torch.randn(5, 45).tolist(),
                "label": i % 4,
                "context": [],
            }
        )
    dataset = tf.WindowDataset(windows, context_dim=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    metrics = tf.evaluate(model, loader, torch.device("cpu"))

    # Reproduz a mesma avaliacao amostra-a-amostra via IntentionPredictor
    # diretamente, e confere que evaluate() bate com o preditor completo.
    from predict import IntentionPredictor

    predictor = IntentionPredictor(model=model)
    expected_preds = []
    for pose, label, context in loader:
        _, batch_intention = predictor.predict(pose, restrict="ood")
        expected_preds.append(batch_intention.item())

    confusion = metrics.confusion_matrix
    rebuilt_confusion = [[0] * 4 for _ in range(4)]
    for (pose, label, context), pred in zip(loader, expected_preds):
        rebuilt_confusion[label.item()][pred] += 1

    assert confusion == rebuilt_confusion


def test_v1_training_runs_and_produces_metrics_json(tmp_path):
    """V1: context_dim=0, fine-tune curto, checkpoint e metrics.json gerados por seed."""
    dataset_path = tmp_path / "dataset_dim0.json"
    _write_synthetic_dataset(dataset_path, context_dim=0)

    out_dir = tmp_path / "runs_v1"
    tf.main(
        [
            "--variant", "V1",
            "--dataset", str(dataset_path),
            "--context-dim", "0",
            "--init-checkpoint", str(ORIGINAL_CHECKPOINT),
            "--seeds", "0", "1",
            "--epochs", "1",
            "--batch-size", "8",
            "--out-dir", str(out_dir),
        ]
    )

    for seed in (0, 1):
        seed_dir = out_dir / f"seed{seed}"
        assert (seed_dir / "checkpoint.pth").exists()
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        assert metrics["variant"] == "V1"
        assert metrics["context_dim"] == 0
        assert "accuracy" in metrics["metrics"]
        assert "confusion_matrix" in metrics["metrics"]
        assert "ece" in metrics["metrics"]

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["num_seeds"] == 2


def test_v2_requires_per_seed_v1_checkpoints(tmp_path):
    """Secao 6: V2 deve inicializar do checkpoint de V1 pareado por seed, nao de V0."""
    dataset_path = tmp_path / "dataset_dim7.json"
    _write_synthetic_dataset(dataset_path, context_dim=7)

    with pytest.raises(ValueError, match="init-checkpoint-per-seed"):
        tf.main(
            [
                "--variant", "V2",
                "--dataset", str(dataset_path),
                "--context-dim", "7",
                "--seeds", "0",
                "--out-dir", str(tmp_path / "runs_v2"),
            ]
        )


def test_v2_pipeline_end_to_end(tmp_path):
    """V1 -> V2: V2 inicializa do checkpoint de V1 (mesmo seed), context_proj ativo."""
    dataset_dim0 = tmp_path / "dataset_dim0.json"
    dataset_dim7 = tmp_path / "dataset_dim7.json"
    _write_synthetic_dataset(dataset_dim0, context_dim=0)
    _write_synthetic_dataset(dataset_dim7, context_dim=7)

    v1_dir = tmp_path / "runs_v1"
    tf.main(
        [
            "--variant", "V1",
            "--dataset", str(dataset_dim0),
            "--context-dim", "0",
            "--init-checkpoint", str(ORIGINAL_CHECKPOINT),
            "--seeds", "0",
            "--epochs", "1",
            "--batch-size", "8",
            "--out-dir", str(v1_dir),
        ]
    )
    v1_checkpoint = v1_dir / "seed0" / "checkpoint.pth"
    assert v1_checkpoint.exists()

    v2_dir = tmp_path / "runs_v2_dim7"
    tf.main(
        [
            "--variant", "V2",
            "--dataset", str(dataset_dim7),
            "--context-dim", "7",
            "--init-checkpoint-per-seed", str(v1_checkpoint),
            "--seeds", "0",
            "--epochs", "1",
            "--freeze-epochs", "0",
            "--batch-size", "8",
            "--out-dir", str(v2_dir),
        ]
    )
    metrics = json.loads((v2_dir / "seed0" / "metrics.json").read_text())
    assert metrics["context_dim"] == 7

    v2_state = torch.load(v2_dir / "seed0" / "checkpoint.pth", map_location="cpu")
    assert "context_proj.weight" in v2_state


def test_dataset_context_dim_mismatch_rejected(tmp_path):
    dataset_path = tmp_path / "dataset_dim7.json"
    _write_synthetic_dataset(dataset_path, context_dim=7)

    with pytest.raises(ValueError, match="context-dim"):
        tf.main(
            [
                "--variant", "V0",
                "--dataset", str(dataset_path),
                "--context-dim", "0",
                "--init-checkpoint", str(ORIGINAL_CHECKPOINT),
                "--out-dir", str(tmp_path / "runs_mismatch"),
            ]
        )
