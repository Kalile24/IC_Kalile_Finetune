"""Testes de aceite da OS-6 (predict.py / DLinear.py).

Roadmap, Secao 8, linha da OS-6:
    "Patch em DLinear.py e predict.py: camada context_proj residual e
    argumento opcional de contexto em predict(). Aceite: teste de
    retrocompatibilidade numerica."

1. test_predict_context_dim_zero_matches_original: com context_dim=0 (ou
   context=None), IntentionPredictor.predict() desta copia produz a mesma
   predicao (pred_traj, batch_intention) que o IntentionPredictor original,
   para o mesmo checkpoint e a mesma entrada.
2. test_predict_context_zero_init_does_not_perturb_output: um
   IntentionPredictor com context_dim>0, carregando o checkpoint original
   (sem context_proj) via strict=False, produz a mesma predicao do original
   independente do vetor de contexto passado - o zero-init garante que a
   injecao de contexto comeca como no-op.
3. test_predict_context_shape_broadcast: um vetor de contexto 1D
   (context_dim,) e aceito e produz o mesmo resultado que a versao ja
   com batch dim (1, context_dim).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

FINETUNE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FINETUNE_DIR))

import predict as local_predict  # noqa: E402

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


def _load_original_predictor_class():
    """Importa IntentionPredictor do repositorio fonte, sem modifica-lo."""
    sys.path.insert(0, str(ORIGINAL_TRAJ_INTENTION))
    import predict as original_predict  # noqa: E402

    return original_predict.IntentionPredictor


def test_predict_context_dim_zero_matches_original():
    """Aceite principal: context_dim=0 == IntentionPredictor original bit a bit."""
    OriginalPredictor = _load_original_predictor_class()

    original = OriginalPredictor(ckpt_path=str(ORIGINAL_CHECKPOINT))
    local = local_predict.IntentionPredictor(
        ckpt_path=str(ORIGINAL_CHECKPOINT), context_dim=0
    )

    torch.manual_seed(0)
    poses = torch.randn(1, 5, 45)

    orig_traj, orig_intention = original.predict(poses, restrict="no")
    local_traj, local_intention = local.predict(poses, restrict="no", context=None)

    assert torch.allclose(orig_traj, local_traj, atol=1e-7)
    assert torch.equal(orig_intention, local_intention)


def test_predict_context_zero_init_does_not_perturb_output():
    """context_proj zero-inicializado: IntentionPredictor com contexto ainda
    reproduz a predicao original, para qualquer vetor de contexto, antes do
    fine-tuning."""
    OriginalPredictor = _load_original_predictor_class()

    original = OriginalPredictor(ckpt_path=str(ORIGINAL_CHECKPOINT))

    torch.manual_seed(1)
    poses = torch.randn(1, 5, 45)
    orig_traj, orig_intention = original.predict(poses, restrict="no")

    for context_dim in (7, 10):
        local = local_predict.IntentionPredictor(
            ckpt_path=str(ORIGINAL_CHECKPOINT), context_dim=context_dim
        )
        context = torch.randn(context_dim)
        local_traj, local_intention = local.predict(poses, restrict="no", context=context)

        assert torch.allclose(orig_traj, local_traj, atol=1e-7), (
            f"context_dim={context_dim}: zero-init context_proj must not perturb "
            "the inherited checkpoint's prediction"
        )
        assert torch.equal(orig_intention, local_intention)


def test_predict_context_shape_broadcast():
    """Contexto 1D (context_dim,) e aceito e equivalente a (1, context_dim)."""
    local = local_predict.IntentionPredictor(
        ckpt_path=str(ORIGINAL_CHECKPOINT), context_dim=7
    )

    torch.manual_seed(2)
    poses = torch.randn(1, 5, 45)
    context_1d = torch.randn(7)
    context_2d = context_1d.unsqueeze(0)

    traj_1d, intention_1d = local.predict(poses, restrict="no", context=context_1d)
    traj_2d, intention_2d = local.predict(poses, restrict="no", context=context_2d)

    assert torch.allclose(traj_1d, traj_2d, atol=1e-7)
    assert torch.equal(intention_1d, intention_2d)


def test_predict_context_accepts_plain_list():
    """context pode ser passado como lista Python (nao so tensor)."""
    local = local_predict.IntentionPredictor(
        ckpt_path=str(ORIGINAL_CHECKPOINT), context_dim=7
    )

    torch.manual_seed(3)
    poses = torch.randn(1, 5, 45)
    context_list = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    traj, intention = local.predict(poses, restrict="no", context=context_list)
    assert traj.shape == (1, 5, 45)
    assert intention.shape == (1,)
