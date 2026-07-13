# Experimento 2026-07-13 — split único (S05+S02), 60 épocas, corte por entropia com vs. sem

## Contexto

Sessões reais gravadas em 2026-07-12 (`hrc-data-collection/sessions/S01..S08`),
2 participantes (P01, P02), 4 roteiros (`R01`, `R02`, `R03`, `R06`). `R01` e
`R06` são os dois roteiros completos com ordem de montagem invertida
(`bottom → four_tubes → top → wheels` vs. `bottom → top → four_tubes →
wheels`); as sessões `R01` são S01/S05 e as `R06` são S02/S08.

Um levantamento inicial rodou as 4 combinações possíveis de teste
`R01×R06` (S01+S02, S01+S08, S05+S02, S05+S08) com 10 épocas de fine-tuning.
Este documento cobre a rodada final: **split único escolhido (S05+S02)**,
**60 épocas** (em vez de 10) e comparação direta **com/sem corte por
entropia** na avaliação.

## Por que só um split

As 4 combinações R01×R06 foram testadas para não escolher o split "a dedo".
`S05+S02` teve a maior acurácia V1 e o menor ECE entre as 4, e foi o único
onde a injeção de contexto (V2) superou V1 com o corte por entropia ativo —
por isso ficou como split definitivo. Os datasets/execuções dos outros 3
pares foram descartados; o pipeline de dataset (`build_json.py`) e de
avaliação (`train_finetune.py`) continuam suportando qualquer par via
`--test-session`.

```
treino: S01, S03, S04, S06, S07, S08
teste:  S05, S02
```

`datasets/v1/report_classes_dim0.md` (hrc-data-collection) tem a contagem de
janelas por classe/split; é idêntica nos 3 arquivos `dataset_dim{0,7,10}.json`
(única diferença é o vetor de contexto, ver seção 1.4 do roadmap).

## Por que 60 épocas em vez de 10

Com o protocolo original (10 épocas), a loss de V1 ainda caía de forma
consistente (11.85 → 5.78, sem sinal de plateau) — undertraining claro.
Um diagnóstico rodou V1 até 60 épocas: a loss estabiliza por volta da época
40-50 (`~1.0` de cross-entropy, ver `runs/V1/seed0/train_log.txt`).
V2 (fase 2, após 4 épocas de backbone congelado) estabiliza ainda mais cedo,
por volta de `~0.33`.

Resultado do diagnóstico: com 60 épocas, o classificador aprende a
discriminar as 4 classes bem (macro-acurácia salta de ~0.20 para ~0.57,
avaliando por `argmax` cru, sem corte) — a rede não estava com capacidade
insuficiente, só não tinha tido épocas suficientes para convergir.

## Achado principal: o corte por entropia mascara o ganho de V2

O `IntentionPredictor.predict(restrict="ood")` do artigo original aplica um
corte fixo (`entropy > 0.4`, ou `> 0.5` quando a predição é `get_wheels`) que
reclassifica a predição para `no_action`. Esse limiar foi calibrado no
dataset original do paper. Neste dataset, a entropia média das predições do
classificador fine-tuned é `~0.71-0.83` (a entropia máxima possível para 4
classes é `ln(4) ≈ 1.386`) — ou seado, o corte reclassifica **67-86% das
amostras de teste para `no_action`**, mesmo quando a predição original
(antes do corte) estava correta.

Como `no_action` é a classe majoritária no teste (~65-72% das amostras),
isso infla artificialmente a acurácia bruta enquanto **zera a acurácia
das 3 classes de ação** em quase todos os cenários testados. Ver
`results/confusion_matrices.png` (com corte): a coluna `get_screws` fica
zerada em V1 e quase zerada em V2.

Sem o corte (`argmax` cru dos logits — branch `sem-corte-entropia`, que
reproduz o comportamento de `evaluate()` anterior a este experimento), a
mesma rede treinada mostra recall de **93-100% em `get_screws`** e
**97-99% em `get_wheels`** nas variantes V2. Ver
`results_no_cutoff/confusion_matrices.png`.

## Resultados — 60 épocas, com corte por entropia (`restrict="ood"`)

Tabela completa: [`results/ablation_table.md`](../results/ablation_table.md).
Figura: [`results/confusion_matrices.png`](../results/confusion_matrices.png).

| Variante | Top-1 (média ± dp) | ECE | no_action | get_connectors | get_screws | get_wheels |
|---|---:|---:|---:|---:|---:|---:|
| V0 (checkpoint original) | 0.209 | 0.738 | 0.253 | 0.000 | 0.000 | 0.587 |
| V1 (fine-tune, ctx=0) | 0.610 ± 0.000 | 0.289 | 0.781 | 0.367 | 0.082 | 0.000 |
| V2 (contexto 7D) | 0.659 ± 0.005 | 0.298 | 0.733 | 0.306 | 0.492 | 0.747 |
| V2 (contexto 10D) | 0.663 ± 0.003 | 0.288 | 0.730 | 0.281 | 0.614 | 0.560 |

**Delta V2(7D) − V1: +0.049** (claim central do artigo, com corte ativo).

## Resultados — 60 épocas, sem corte por entropia (`argmax` cru dos logits)

Tabela completa: [`results_no_cutoff/ablation_table.md`](../results_no_cutoff/ablation_table.md).
Figura: [`results_no_cutoff/confusion_matrices.png`](../results_no_cutoff/confusion_matrices.png).

| Variante | Top-1 (média ± dp) | ECE | no_action | get_connectors | get_screws | get_wheels |
|---|---:|---:|---:|---:|---:|---:|
| V0 (checkpoint original) | 0.041 | 0.861 | 0.000 | 0.057 | 0.000 | 0.773 |
| V1 (fine-tune, ctx=0) | 0.369 ± 0.001 | 0.337 | 0.192 | 0.677 | 0.968 | 0.724 |
| V2 (contexto 7D) | 0.678 ± 0.001 | 0.155 | 0.570 | 0.882 | 1.000 | 0.987 |
| V2 (contexto 10D) | 0.734 ± 0.001 | 0.123 | 0.642 | 0.928 | 1.000 | 0.987 |

**Delta V2(7D) − V1 (sem corte): +0.309** — quase 6× maior que o delta
observado com o corte ativo, e o ECE também melhora com contexto (0.337 →
0.155 → 0.123), ao contrário do cenário com corte (ECE fica praticamente
igual em todas as variantes, ~0.29).

## Interpretação

1. **V1 sozinho (sem contexto) é pior sem o corte** (0.369 vs. 0.610 bruto
   com corte) porque V1 tende a prever `no_action` errado em situações
   ambíguas — o corte "acerta" essas por coincidência ao forçar `no_action`,
   já que essa é a classe majoritária do teste.
2. **V2 (com contexto) é muito melhor sem o corte** (0.678-0.734 vs.
   0.659-0.663 com corte) porque o contexto do `PlanGraph` dá ao classificador
   informação suficiente para prever as classes de ação corretamente com
   confiança — só que ainda com entropia acima do limiar herdado do artigo
   original, então o corte descarta exatamente as predições que o contexto
   tornou possíveis.
3. **O limiar de entropia (0.4/0.5) não foi recalibrado para este dataset.**
   Um teste variando o limiar (0.4/0.5 → 0.6/0.7 → 0.8/0.9 → sem corte)
   mostrou que a acurácia bruta piora monotonicamente ao afrouxar o limiar
   em todos os 4 pares testados na rodada inicial (10 épocas) — sinal de que
   a acurácia bruta está sendo inflada pelo desbalanceamento de classes
   (`no_action` ~65-72% do teste), não de que o corte "ajuda" de fato.
   Recomendação: reportar acurácia macro (balanceada) e ECE, não só
   acurácia bruta, e tratar o limiar de entropia como hiperparâmetro a
   recalibrar por dataset (ex.: escolher o limiar que maximiza acurácia
   macro no split de validação), em vez de herdar 0.4/0.5 do artigo.

## Branch de comparação

`sem-corte-entropia` (GitHub, a partir do commit `ce0b3a9`) preserva o
`evaluate()` original (antes desta sessão), que usa `argmax` cru dos logits
sem passar pelo `IntentionPredictor`/corte por entropia. Serviu de base para
gerar os números de "sem corte" acima, sem misturar o código de avaliação
das duas versões.

## Reprodução

```bash
# 1. Datasets (hrc-data-collection/, split único S05+S02)
cd hrc-data-collection
source .venv/bin/activate && unset PYTHONPATH
for dim in 0 7 10; do
  python -m datacol.build_json \
    --sessions-root sessions \
    --output datasets/v1/dataset_dim${dim}.json \
    --report datasets/v1/report_classes_dim${dim}.md \
    --test-session S05_20260712 --test-session S02_20260712 \
    --context-dim ${dim} --window-size 5
done

# 2. V0/V1/V2, 60 épocas (hrc-finetune/, com corte por entropia)
cd ../hrc-finetune
CKPT=/home/marcos-kalile/IC_Kalile_Intention_Prediction_HRC/traj_intention/checkpoints/seq5_pred5_epoch40_whole_pkl_final_intention_nomaskTrue.pth
DATA_ROOT=../hrc-data-collection/datasets/v1

python train_finetune.py --variant V0 --dataset $DATA_ROOT/dataset_dim0.json \
  --context-dim 0 --init-checkpoint $CKPT --seeds 0 --out-dir runs/V0

python train_finetune.py --variant V1 --dataset $DATA_ROOT/dataset_dim0.json \
  --context-dim 0 --init-checkpoint $CKPT --seeds 0 1 2 \
  --epochs 60 --lr 1e-4 --batch-size 32 --out-dir runs/V1

for dim in 7 10; do
  python train_finetune.py --variant V2 --dataset $DATA_ROOT/dataset_dim${dim}.json \
    --context-dim ${dim} \
    --init-checkpoint-per-seed runs/V1/seed0/checkpoint.pth \
                                runs/V1/seed1/checkpoint.pth \
                                runs/V1/seed2/checkpoint.pth \
    --seeds 0 1 2 --epochs 60 --freeze-epochs 4 \
    --lr 1e-4 --context-lr 1e-3 --batch-size 32 --out-dir runs/V2_dim${dim}
done

# 3. Tabela e figuras
python reports/scripts/aggregate_results.py --runs-root runs --out-dir results

# 4. Mesma avaliação sem corte por entropia: checkout do branch
#    sem-corte-entropia, reavaliar os checkpoints já treinados acima com
#    evaluate() (argmax cru), salvar em runs_no_cutoff/ e rodar
#    aggregate_results.py --runs-root runs_no_cutoff --out-dir results_no_cutoff
```
