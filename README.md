# IC Kalile — Fine-tune com Injeção de Contexto de Tarefa

Este repositório contém o pipeline de fine-tuning e avaliação usado na
Iniciação Científica (PIBIC/CNPq, IME/LIARC) sobre predição de intenção em
colaboração humano-robô (HRC). O objetivo é medir o ganho de acurácia obtido
ao injetar um vetor de contexto de tarefa (progresso da montagem) no
preditor de intenção `DLinear` de
[Yu et al., arXiv:2411.15711](https://arxiv.org/abs/2411.15711), comparando
três variantes de modelo:

| Variante | Descrição |
|---|---|
| **V0** | Checkpoint original, sem fine-tuning — mede o gap entre webcam e dados originais |
| **V1** | Fine-tune sem contexto — mede o ganho só do fine-tuning |
| **V2** | Fine-tune com contexto, partindo de V1 — resultado principal (delta V2 − V1) |

Este repositório trabalha em conjunto com o
[`hrc-data-collection`](https://github.com/Kalile24/IC_Kalile_Dataset), que
grava, anota e consolida as sessões usadas para gerar os datasets
consumidos aqui, e com o
[`IC_Kalile_Intention_Prediction_HRC`](https://github.com/Kalile24/IC_Kalile_Intention_Prediction_HRC),
fonte do checkpoint original e da arquitetura `DLinear` de referência.

## Estrutura

```text
DLinear.py              Model_FinalIntention/Model_FinalTraj (DLinear), com
                         injeção de contexto opcional (context_dim=0 é
                         idêntico ao modelo original — retrocompatibilidade
                         testada em tests/test_train_finetune.py)
predict.py               IntentionPredictor: carrega checkpoint e roda
                         inferência, com argumento opcional de contexto e
                         o corte por entropia do preditor completo do
                         artigo (restrict="ood"); aceita model= para reusar
                         um modelo já carregado em memória
plan_sim.py               PlanGraph: simula o progresso da montagem e as
                         transições de estágio/contexto, sem depender de ROS
train_finetune.py         Treino e avaliação das variantes V0/V1/V2
run_webcam_context.py     Inferência ao vivo (webcam + MediaPipe Pose) com
                         contexto do PlanGraph injetado
tests/                    Testes automatizados (pytest, 18 no total)
reports/                  Guia de execução, relatório do experimento real e
                         script de agregação de resultados
runs/, results/           Checkpoints e métricas com o corte por entropia
                         ativo (comportamento completo do artigo)
runs_no_cutoff/,
results_no_cutoff/        Os mesmos checkpoints, reavaliados sem o corte por
                         entropia — ver "Achado principal" abaixo
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
  --epochs 60 --lr 1e-4 --batch-size 32 --out-dir runs/V1

# V2: fine-tune com contexto (7D), partindo dos checkpoints de V1
python train_finetune.py --variant V2 \
  --dataset $DS/dataset_dim7.json --context-dim 7 \
  --init-checkpoint-per-seed runs/V1/seed0/checkpoint.pth \
                              runs/V1/seed1/checkpoint.pth \
                              runs/V1/seed2/checkpoint.pth \
  --seeds 0 1 2 --epochs 60 --freeze-epochs 4 \
  --lr 1e-4 --context-lr 1e-3 --batch-size 32 --out-dir runs/V2_dim7
```

Cada rodada grava checkpoint + `metrics.json` por seed, mais `summary.json`
com média ± desvio-padrão.

> **Épocas:** use `--epochs 60`, não 10. Com 10 épocas a perda de treino
> ainda cai de forma consistente (sub-treino); ela estabiliza por volta da
> época 40–50. Ver
> [`reports/experimento_2026-07-13_split_unico_epochs60.md`](reports/experimento_2026-07-13_split_unico_epochs60.md)
> para a comparação numérica.

### Avaliação: com e sem corte por entropia

`evaluate()` em `train_finetune.py` avalia por padrão através do
`IntentionPredictor` completo (`restrict="ood"`), que reclassifica
predições de baixa confiança para `no_action` — o comportamento do artigo
original. O branch [`sem-corte-entropia`](../../tree/sem-corte-entropia)
preserva o `evaluate()` anterior (argmax cru dos logits, sem esse filtro),
para comparação controlada sobre os mesmos checkpoints.

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

O guia completo de execução (gravação de sessões, consolidação de
datasets, treino e agregação, passo a passo) está em
[`reports/GUIA_FINAL.md`](reports/GUIA_FINAL.md).

## Testes

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/
```

O bloqueio de plugins evita que plugins pytest de um ambiente ROS global
contaminem este repositório independente.

## Achado principal (experimento de 2026-07-13)

O corte por entropia do artigo original (limiares `0,4`/`0,5`, calibrados
no dataset original — câmera de profundidade, detecção hierárquica de
pessoa, suavização temporal de keypoints, nenhum presente aqui) reclassifica
67–86% das predições de teste para `no_action` neste dataset de webcam,
mascarando o ganho real do vetor de contexto:

| Cenário | Delta V2(7D) − V1 (acurácia top-1) |
|---|---:|
| Com corte por entropia (comportamento do artigo) | +4,9 p.p. |
| Sem corte (rede neural isolada) | **+30,9 p.p.** |

Detalhes, matrizes de confusão e a comparação com o artigo original estão
em [`reports/experimento_2026-07-13_split_unico_epochs60.md`](reports/experimento_2026-07-13_split_unico_epochs60.md).

## Como ler os resultados

- Diagonal da matriz de confusão = recall por classe (cada linha soma 1).
- V0 → V1: melhora mede a adaptação de domínio (fine-tuning puro).
- V1 → V2: melhora adicional é o efeito da injeção de contexto — resultado
  central do trabalho. Compare sempre nos dois modos (com e sem corte por
  entropia) antes de tirar conclusões sobre o tamanho desse efeito.
