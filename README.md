# IC Kalile — Fine-tune com Injeção de Contexto de Tarefa

Este repositório contém o pipeline de fine-tuning e avaliação usado na minha
Iniciação Científica sobre predição de intenção em colaboração humano-robô
(HRC). O objetivo é medir o ganho de acurácia obtido ao injetar um vetor de
contexto de tarefa (estado do plano de montagem) no preditor de intenção
`DLinear`, comparando três variantes de modelo:

| Variante | Descrição |
|---|---|
| **V0** | Checkpoint original, sem fine-tuning — mede o gap entre webcam e dados originais |
| **V1** | Fine-tune sem contexto — mede o ganho só do fine-tuning |
| **V2** | Fine-tune com contexto, partindo de V1 — resultado principal (delta V2 − V1) |

Este repositório trabalha em conjunto com o [`hrc-data-collection`](https://github.com/Kalile24/IC_Kalile_Dataset),
que grava, anota e consolida as sessões usadas para gerar os datasets
consumidos aqui.

## Estrutura

```
DLinear.py              Modelo Model_FinalIntention/Model_FinalTraj (DLinear),
                         com injeção de contexto opcional (context_dim=0 é
                         idêntico ao modelo original)
predict.py               IntentionPredictor: carrega checkpoint e roda inferência,
                         com argumento opcional de contexto
plan_sim.py               PlanGraph: simula o estado do plano de montagem e as
                         transições de estágio/contexto, sem depender de ROS
train_finetune.py         Script de treino para as variantes V0/V1/V2
run_webcam_context.py     Pipeline de inferência ao vivo (webcam + MediaPipe
                         Pose) com contexto do PlanGraph injetado
tests/                    Testes automatizados (pytest)
reports/                  Relatório final (PDF/LaTeX), guia de execução e
                         script de agregação de resultados
```

## Requisitos

- Python 3.9+
- `torch`, `opencv-python`, `numpy`
- Para o pipeline de webcam: `mediapipe`
- Para agregação de resultados: `scikit-learn`, `matplotlib`

```bash
pip install torch opencv-python numpy mediapipe scikit-learn matplotlib
```

## Uso

### Treinar as variantes V0/V1/V2

```bash
CKPT=/caminho/para/checkpoint_original.pth
DS=/caminho/para/datasets/v1

# V0: apenas avalia o checkpoint original
python train_finetune.py --variant V0 \
  --dataset $DS/dataset_dim0.json --context-dim 0 \
  --init-checkpoint $CKPT --out-dir runs/V0

# V1: fine-tune sem contexto, 3 sementes
python train_finetune.py --variant V1 \
  --dataset $DS/dataset_dim0.json --context-dim 0 \
  --init-checkpoint $CKPT --seeds 0 1 2 \
  --epochs 10 --lr 1e-4 --batch-size 32 --out-dir runs/V1

# V2: fine-tune com contexto (7D), partindo dos checkpoints de V1
python train_finetune.py --variant V2 \
  --dataset $DS/dataset_dim7.json --context-dim 7 \
  --init-checkpoint-per-seed runs/V1/seed0/checkpoint.pth \
                              runs/V1/seed1/checkpoint.pth \
                              runs/V1/seed2/checkpoint.pth \
  --seeds 0 1 2 --epochs 10 --freeze-epochs 4 \
  --lr 1e-4 --context-lr 1e-3 --batch-size 32 --out-dir runs/V2_dim7
```

Cada rodada grava checkpoint + `metrics.json` por seed, mais `summary.json`
com média ± desvio-padrão.

### Agregar resultados finais

```bash
pip install scikit-learn matplotlib -q
python reports/scripts/aggregate_results.py --runs-root runs --out-dir results
```

Gera `results/ablation_table.md` (tabela V0/V1/V2 com acurácia, ECE e delta
V2−V1) e `results/confusion_matrices.png`.

### Inferência ao vivo (webcam)

```bash
# Sem contexto
python run_webcam_context.py --show --task webcam001

# Com contexto do PlanGraph (7D)
python run_webcam_context.py --show --task webcam001 --context-dim 7

# Replay de um .pkl gravado
python run_webcam_context.py --diag --replay caminho/para/sessao.pkl
```

O guia completo de execução (gravação de sessões, consolidação de datasets,
treino e agregação, passo a passo) está em [`reports/GUIA_FINAL.md`](reports/GUIA_FINAL.md)
e na versão detalhada em PDF, [`reports/relatorio_final_ic.pdf`](reports/relatorio_final_ic.pdf).

## Testes

```bash
pytest tests/
```

## Como ler os resultados

- Diagonal da matriz de confusão = acurácia por classe.
- V0 → V1: melhora mede a adaptação de domínio (fine-tuning puro).
- V1 → V2: melhora adicional é o efeito da injeção de contexto — resultado
  central do trabalho.
