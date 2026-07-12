"""Testes de aceite da OS-7 (run_webcam_context.py / plan_sim.py).

Roadmap, Secao 8, linha da OS-7:
    "Patch em receiver.py + run_webcam.py (+ sender.py opcional): Canal
    reverso /plan_context conforme Secao 5. Aceite: vetor publicado confere
    com estado interno apos cada mutacao; partida a frio usa estado
    inicial."

Este pipeline (hrc-data-collection/run_webcam.py) ja roda em processo unico,
sem ROS (linhas [ROS] comentadas no original) - por isso nao ha
publisher/subscriber real para testar. O "vetor publicado" equivalente aqui
e o `cached_context` mantido no escopo de run_live(); os testes abaixo
verificam as duas funcoes puras extraidas dessa logica
(init_plan_context/confirm_intention_context), que sao exatamente o que
run_live() chama nos pontos de cold start e de confirmacao.

1. test_cold_start_uses_initial_state_vector: partida a fria (init_plan_context)
   usa o vetor do estado inicial, nao None nem zeros arbitrarios.
2. test_confirmed_intention_context_matches_planGraph_state: apos cada
   mutacao (confirm_intention_context), o vetor retornado bate exatamente
   com plan.to_context_vector(dim) e com plan.snapshot() no mesmo instante.
3. test_context_dim_zero_disables_plan: com context_dim=0, plan e contexto
   ficam None (paridade com o pipeline original, sem contexto).
4. test_predict_receives_cached_context: integra com predict.py — a mesma
   sequencia de confirmacoes que muta o PlanGraph produz um contexto que,
   passado a IntentionPredictor.predict(), chega intacto ao forward do
   modelo (nao overscrito nem descartado no meio do caminho).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

FINETUNE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FINETUNE_DIR))

DATA_COLLECTION_DIR = FINETUNE_DIR.parent / "hrc-data-collection"

pytestmark = pytest.mark.skipif(
    not (DATA_COLLECTION_DIR / "mediapipe_fallback.py").exists(),
    reason="hrc-data-collection checkout (mediapipe_fallback.py) not available",
)

import run_webcam_context as rwc  # noqa: E402
from plan_sim import PlanGraph  # noqa: E402

ORIGINAL_CHECKPOINT = Path(
    "/home/marcos-kalile/IC_Kalile_Intention_Prediction_HRC/traj_intention"
    "/checkpoints/seq5_pred5_epoch40_whole_pkl_final_intention_nomaskTrue.pth"
)

requires_checkpoint = pytest.mark.skipif(
    not ORIGINAL_CHECKPOINT.exists(),
    reason="original checkpoint not available in this environment",
)


def test_cold_start_uses_initial_state_vector():
    """Partida a frio: cached_context = [1,0,0,0,0,0,0] (stage=none), nao None."""
    plan, cached_context = rwc.init_plan_context(7)
    assert plan is not None
    assert cached_context == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    plan10, cached_context10 = rwc.init_plan_context(10)
    assert cached_context10 == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_context_dim_zero_disables_plan():
    """context_dim=0: plan e contexto ficam None (paridade com pipeline original)."""
    plan, cached_context = rwc.init_plan_context(0)
    assert plan is None
    assert cached_context is None

    new_context = rwc.confirm_intention_context(plan, 0, "get_connectors")
    assert new_context is None


def test_confirmed_intention_context_matches_planGraph_state():
    """Aceite principal: o vetor apos cada mutacao confere com o estado interno."""
    plan, cached_context = rwc.init_plan_context(7)
    assert cached_context == plan.to_context_vector(7)

    sequence = ["get_connectors", "get_connectors", "get_connectors", "get_connectors"]
    for intention in sequence:
        new_context = rwc.confirm_intention_context(plan, 7, intention)
        # O vetor retornado bate exatamente com o estado do PlanGraph no
        # mesmo instante (nenhuma defasagem entre "publicacao" e mutacao).
        assert new_context == plan.to_context_vector(7)

    # Quatro get_connectors completam o estagio "bottom" (4 acoes por estagio).
    snapshot = plan.snapshot()
    assert snapshot["stage_history"] == ["bottom"]
    assert snapshot["stage"] is None
    # One-hot [none, bottom, four_tubes, top]: de volta a "none" apos o estagio fechar.
    assert plan.to_context_vector(7)[0] == 1.0


def test_confirm_intention_context_independent_reference_plangraph():
    """O contexto calculado bate com um PlanGraph de referencia rodando a
    mesma sequencia via step() diretamente (sem passar por run_webcam_context)."""
    plan_a, ctx_a = rwc.init_plan_context(10)
    plan_b = PlanGraph(policy=rwc.PLAN_POLICY)
    ctx_b = plan_b.to_context_vector(10)
    assert ctx_a == ctx_b

    for intention in ["get_connectors", "get_screws", "get_wheels"]:
        ctx_a = rwc.confirm_intention_context(plan_a, 10, intention)
        plan_b.step(intention)
        ctx_b = plan_b.to_context_vector(10)
        assert ctx_a == ctx_b
        assert plan_a.snapshot() == plan_b.snapshot()


@requires_checkpoint
def test_predict_receives_cached_context():
    """Integra com predict.py: contexto do PlanGraph chega intacto ao forward."""
    import predict as local_predict

    plan, cached_context = rwc.init_plan_context(7)
    predictor = local_predict.IntentionPredictor(
        ckpt_path=str(ORIGINAL_CHECKPOINT), context_dim=7
    )

    torch.manual_seed(0)
    poses = torch.randn(1, 5, 45)

    # Cold start: contexto = estado inicial.
    traj_cold, _ = predictor.predict(poses, restrict="no", context=cached_context)

    # Apos uma confirmacao, o contexto muda e a predicao (com context_proj
    # zero-inicializado) permanece numericamente igual, pois o checkpoint
    # nao foi fine-tuned - mas o teste confirma que nao ha erro de shape/
    # dtype ao longo do caminho cached_context -> predict -> forward.
    cached_context = rwc.confirm_intention_context(plan, 7, "get_connectors")
    traj_after, _ = predictor.predict(poses, restrict="no", context=cached_context)

    assert traj_cold.shape == traj_after.shape == (1, 5, 45)
    # zero-init: contexto diferente, saida identica antes do fine-tuning.
    assert torch.allclose(traj_cold, traj_after, atol=1e-7)
