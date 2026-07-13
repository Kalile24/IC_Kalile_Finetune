# Guia Final de Execução — IC

Injeção de Contexto de Tarefa no Preditor de Intenção

> Todo o código (OS-1 a OS-7) já está pronto e testado (18 testes em
> `hrc-finetune/tests/`, 52 em `hrc-data-collection/tests/`).
> Este guia descreve o procedimento completo de 4 blocos, nesta ordem:
> **(1) gravar e anotar sessões, (2) consolidar 3 datasets, (3) treinar
> V0/V1/V2, (4) gerar tabela e figuras finais.**
>
> A primeira rodada real já foi executada: 8 sessões, 2 participantes,
> split de teste fixado em `S05_20260712` + `S02_20260712`. Resultados e
> discussão em
> [`experimento_2026-07-13_split_unico_epochs60.md`](experimento_2026-07-13_split_unico_epochs60.md).
> Este `.md` permanece como referência operacional para rodadas futuras de
> coleta (mais sessões, outro split).

Ambiente: conda `hrc` (`source /home/marcos-kalile/anaconda3/bin/activate hrc`).
Coleta e consolidação rodam em `hrc-data-collection/`; treino e agregação em `hrc-finetune/`.

---

## Visão geral dos 4 blocos

| Bloco | O que fazer | Ferramenta | Tempo estimado |
|---|---|---|---|
| 1 | Gravar e anotar ~12 sessões | `capture_session`, `annotate_pkl` | 1–2 dias |
| 2 | Consolidar 3 datasets de contexto | `build_json` | 15 min |
| 3 | Treinar V0, V1, V2 | `train_finetune.py` | algumas horas (CPU) |
| 4 | Gerar tabela e matrizes de confusão | `aggregate_results.py` | 5 min |

---

## Bloco 1 — Gravar e anotar as sessões

### Bancada (montar uma vez, não mexer mais)

- 8 conectores, 12 parafusos, 4 rodas, 8 tubos curtos (~12 cm), 4 tubos longos (~25 cm);
- conectores à direita; parafusos à esquerda perto; rodas à esquerda atrás dos parafusos; gabarito ao centro; marca de repouso na borda;
- 40–50 cm entre zonas.

**Não mude entre sessões:** câmera, posição das zonas, cadeira, iluminação — fixas do início ao fim. Só variam ordem de pegas, ritmo e postura de repouso.

### As 6 rotas canônicas

Vêm de `Rotas_e_Vetor_de_Contexto.pdf` (política `proxy_graph`, a mesma que `plan_sim.py` já usa). Passos e contexto 7D já são a saída real do simulador, não estimativa.

| Rota | Papel | Sequência | Passos |
|---|---|---|---:|
| **R01** | Tarefa completa (canônica) | `bottom → four_tubes → top → wheels`. Fecha com contexto `[1000\|1.00,1.00,1.00]`. Rota de referência. | 29 |
| R02 | Estágio `bottom` isolado | Só a primeira parte de R01 (4 conectores + 4 parafusos de `bottom`). Rota mínima de validação. | 8 |
| R03 | `bottom → four_tubes` | Prefixo de R01 até o fim de `four_tubes`, sem `top`/`wheels`. | 17 |
| R04 | Retomada via `--stageI_done` | Começa direto em `four_tubes` (bottom já pronto), segue `top → wheels`. | 21 |
| R05 | Completa com `no_action` intercalado | Mesma sequência de R01, com pausas de `no_action` em pontos de contexto igual — engorda essa classe sem inventar contexto novo. | 32 |
| **R06** | Ordem alternativa | `bottom → top → four_tubes → wheels` (troca posição de `top`/`four_tubes` vs. R01). Também válida no grafo. | 29 |

> A tabela abaixo resume as 6 rotas; o roteiro gesto a gesto de cada uma foi
> usado para gravar as sessões reais e não está mais reproduzido aqui.

**Como usar as rotas:**

- **R01 primeiro, sempre**: grave e anote só ela, rode o Bloco 2 com `--allow-empty-split` para validar a cadeia inteira antes de investir nas demais.
- **Diversidade estrutural** (robustece o contexto contra viés de ordem): alterne R01 e R06 nas repetições de treino.
- **Diversidade de trajetória** (mesmo grafo, mesmo contexto, esqueleto diferente): use R05 para gerar `no_action` com o mesmo par (contexto, próxima intenção) de R01, variando só a execução física.
- **R02/R03/R04 são rotas curtas**: úteis para engordar classes específicas sem repetir a tarefa inteira, quando faltar janela de uma classe pontual.
- Recomendação do documento de rotas: execute R01/R06 com variação natural (ritmo, postura) e **repita cada uma 8–12 vezes** para densificar amostras por par (intenção, estágio).

### Quantas sessões gravar

| Sessões | O que gravar | Split |
|---|---|---|
| 1.ª | R01 (piloto — valida a cadeia antes de continuar) | treino |
| 2.ª–5.ª | Alterne R01/R06 (2 de cada), ritmo e postura naturais entre repetições | treino |
| 6.ª–8.ª | R05 (completa com `no_action` intercalado), variando ritmo | treino |
| 9.ª | R02, R03 ou R04 — só a que estiver mais fraca após o Bloco 2 | treino |
| **10.ª** | **R01** completa, ritmo normal. Sessão **inteira** reservada para teste — não editar depois de gravar. | **teste** |
| **11.ª** | **R06** completa, ritmo normal. Segunda sessão de teste. | **teste** |
| 12.ª (buffer) | **Só grave se** o relatório do Bloco 2 mostrar alguma classe com <120 janelas. Repita a rota que cobre essa classe. | a definir |

### Como gravar cada sessão

Repita para cada uma das 11–12 sessões, trocando `--script-id` pela rota da linha atual.

```bash
cd /home/marcos-kalile/hrc-data-collection
source .venv/bin/activate

# ID sugerido automaticamente (formato SNN_YYYYMMDD)
python -m datacol.capture_session --suggest-session-id

# Gravar (troque --script-id pela rota desta sessão: R01, R06, R05, ...)
python -m datacol.capture_session \
  --participant P01 --script-id R01 \
  --camera-model "Intelbras WCI 1080p" --camera-distance-m 2.2 \
  --width 1920 --height 1080 --fps 30

# Validar logo em seguida
python -m datacol.capture_session --validate-session sessions/<ID_GERADO>
```

Regra de execução do gesto, sempre: parado no repouso → mão até a zona → pega e deposita → volta ao repouso → pausa breve → próximo gesto. Nunca encadeie dois gestos sem passar pelo repouso. Varie ritmo e postura naturalmente entre repetições da mesma rota.

### Como anotar cada sessão

Anote logo depois de gravar (não acumule).

```bash
python -m datacol.annotate_pkl sessions/<ID_GERADO>
```

Por gesto: `Espaço` pausa/toca → `b` no primeiro quadro do gesto → navegue até o último quadro → tecla da classe (`1`=conectores, `2`=parafusos, `3`=rodas, `i`=ignore). No bloco dos 4 tubos curtos de `four_tubes`: marque tudo como `ignore`, volte ao primeiro quadro do bloco e pressione `f`. Pressione `s` para salvar — o programa recusa se houver lacuna ou sobreposição.

Contagem esperada por sessão R01/R06 (tarefa completa): 8 `get_connectors`, 12 `get_screws`, 4 `get_wheels`, 9 `ignore`, 1 evento `begin_four_tubes`. Rotas curtas (R02/R03/R04) têm contagens proporcionalmente menores — confira contra a coluna "Passos" da tabela de rotas.

**Toda sessão com `four_tubes`** (R01, R03, R04, R05, R06): antes de consolidar, rode:

```bash
python -m datacol.context_replay sessions/<ID> --context-dim 7
```

e confirme visualmente que o estágio muda no quadro certo.

---

## Bloco 2 — Consolidar os 3 datasets

Gere os três JSONs (`context_dim` 0, 7, 10) numa única rodada, sempre com os mesmos IDs de teste:

```bash
cd /home/marcos-kalile/hrc-data-collection
source .venv/bin/activate

for dim in 0 7 10; do
  python -m datacol.build_json \
    --sessions-root sessions \
    --output datasets/v1/dataset_dim${dim}.json \
    --report datasets/v1/report_classes_dim${dim}.md \
    --test-session <ID_SESSAO_10_R01> \
    --test-session <ID_SESSAO_11_R06> \
    --context-dim ${dim} \
    --window-size 5
done
```

Troque `<ID_SESSAO_10_R01>`/`<ID_SESSAO_11_R06>` pelos IDs reais das duas sessões de teste.

### Checklist de sanidade (2 minutos, não pule)

Abra os três `report_classes_dim*.md` e confira:

- [ ] Contagem de janelas por classe é **igual** nos três relatórios.
- [ ] Nenhuma classe com zero janelas em nenhum split.
- [ ] Cada classe com ≥120 janelas (treino+teste). Se faltar → grave a sessão 12 (buffer) focada nessa classe e repita o Bloco 2.
- [ ] `no_action` não passa de ~50% do total.

---

## Bloco 3 — Treinar V0, V1, V2

| Variante | O que é | Para que serve |
|---|---|---|
| V0 | Checkpoint original, sem treinar | mede o gap webcam vs. dados originais |
| V1 | Fine-tune sem contexto | mede o ganho só do fine-tuning |
| V2 | Fine-tune com contexto, partindo de V1 | **resultado principal**: delta V2−V1 |

**Regra fixa:** treine V0 → V1 → V2, nesta ordem. V2 sempre parte do checkpoint de V1 (nunca de V0), e cada seed de V2 parte do **mesmo número de seed** em V1.

> **Épocas:** a primeira rodada usava `--epochs 10`, mas a perda de treino
> ainda caía de forma consistente nesse ponto (sub-treino). A rodada real
> usou `--epochs 60` (a perda estabiliza por volta da época 40–50) — os
> comandos abaixo já refletem isso. Ver
> [`experimento_2026-07-13_split_unico_epochs60.md`](experimento_2026-07-13_split_unico_epochs60.md)
> para a comparação numérica entre 10 e 60 épocas.

```bash
cd /home/marcos-kalile/hrc-finetune
source /home/marcos-kalile/anaconda3/bin/activate hrc

CKPT=/home/marcos-kalile/IC_Kalile_Intention_Prediction_HRC/traj_intention/checkpoints/seq5_pred5_epoch40_whole_pkl_final_intention_nomaskTrue.pth
DS=/home/marcos-kalile/hrc-data-collection/datasets/v1

# --- V0: só avalia o checkpoint original ---
python train_finetune.py --variant V0 \
  --dataset $DS/dataset_dim0.json --context-dim 0 \
  --init-checkpoint $CKPT --out-dir runs/V0

# --- V1: fine-tune sem contexto, 3 sementes ---
python train_finetune.py --variant V1 \
  --dataset $DS/dataset_dim0.json --context-dim 0 \
  --init-checkpoint $CKPT --seeds 0 1 2 \
  --epochs 60 --lr 1e-4 --batch-size 32 --out-dir runs/V1

# --- V2: fine-tune com contexto 7D, partindo de V1 (mesma seed) ---
python train_finetune.py --variant V2 \
  --dataset $DS/dataset_dim7.json --context-dim 7 \
  --init-checkpoint-per-seed runs/V1/seed0/checkpoint.pth \
                              runs/V1/seed1/checkpoint.pth \
                              runs/V1/seed2/checkpoint.pth \
  --seeds 0 1 2 --epochs 60 --freeze-epochs 4 \
  --lr 1e-4 --context-lr 1e-3 --batch-size 32 --out-dir runs/V2_dim7

# --- V2 (10D): ablação (dimensão maior de contexto) ---
python train_finetune.py --variant V2 \
  --dataset $DS/dataset_dim10.json --context-dim 10 \
  --init-checkpoint-per-seed runs/V1/seed0/checkpoint.pth \
                              runs/V1/seed1/checkpoint.pth \
                              runs/V1/seed2/checkpoint.pth \
  --seeds 0 1 2 --epochs 60 --freeze-epochs 4 \
  --lr 1e-4 --context-lr 1e-3 --batch-size 32 --out-dir runs/V2_dim10
```

Cada rodada grava checkpoint + `metrics.json` por seed, mais `summary.json` com média ± desvio-padrão. Se o dataset final tiver poucas janelas por classe, baixe `--batch-size` para 8–16.

> **Corte por entropia:** por padrão, `evaluate()` avalia através do
> `IntentionPredictor` completo (`restrict="ood"`), que reclassifica
> predições de baixa confiança para `no_action`. O branch
> `sem-corte-entropia` preserva o comportamento anterior (avaliação direta
> da rede, sem esse filtro) para comparação — ver a mesma seção do
> experimento citada acima.

> **Por que não há perda de trajetória:** o treino original usa duas perdas (classificação + regressão de trajetória futura). O dataset novo (`build_json.py`) só grava a janela de entrada e o rótulo, não o alvo de trajetória futura, então `train_finetune.py` treina só com a perda de classificação. Isso é proposital: o objetivo é isolar o efeito do contexto na classificação, não reproduzir a tarefa de trajetória do artigo original.

---

## Bloco 4 — Tabela final e matrizes de confusão

```bash
cd /home/marcos-kalile/hrc-finetune
source /home/marcos-kalile/anaconda3/bin/activate hrc
pip install scikit-learn matplotlib -q

python reports/scripts/aggregate_results.py \
  --runs-root runs --out-dir results
```

Gera `results/ablation_table.md` (tabela V0/V1/V2 com acurácia, ECE e o delta V2−V1) e `results/confusion_matrices.png` (uma matriz por variante, lado a lado).

### Como ler o resultado

- Diagonal da matriz = acurácia por classe (cada linha soma 1).
- V0 → V1: melhora mede a adaptação de domínio (fine-tuning puro).
- V1 → V2: melhora adicional é o efeito do contexto — **este é o resultado central do artigo**.
- Metas de referência do roadmap: top-1 ≥ 70% e ECE < 0,08 em V2. Se não bater, o resultado ainda é publicável: o que importa é o delta V2−V1, não o valor absoluto.

---

## Checklist único (do início ao fim)

- [ ] Gravar + anotar a 1.ª sessão (rota R01). Rodar Bloco 2 com `--allow-empty-split` para validar a cadeia.
- [ ] Gravar + anotar as demais 9–10 sessões de treino (rotas R01/R06/R05/R02/R03/R04), anotando cada uma logo após gravar.
- [ ] Gravar + anotar as 2 sessões de teste (rotas R01 e R06, ritmo normal, sessão inteira reservada).
- [ ] Rodar `context_replay` em toda sessão com `four_tubes` (R01, R03, R04, R05, R06).
- [ ] Rodar Bloco 2 completo (3 datasets) com os IDs reais das 2 sessões de teste.
- [ ] Checar os 3 relatórios de classes. Se faltar janela, gravar a sessão 12 (buffer) e repetir.
- [ ] Rodar V0, depois V1 (3 sementes), depois V2 7D (3 sementes pareadas). V2 10D se sobrar tempo.
- [ ] Rodar `aggregate_results.py`.
- [ ] Escrever a seção de resultados do artigo com `ablation_table.md` e `confusion_matrices.png`.

---

## Onde está cada arquivo

- Coleta/anotação/consolidação: `hrc-data-collection/src/datacol/`
- Treino e inferência: `hrc-finetune/{train_finetune,DLinear,predict}.py`
- Testes: `hrc-finetune/tests/`
- Agregação de resultados: `hrc-finetune/reports/scripts/aggregate_results.py`
- Este guia: `hrc-finetune/reports/GUIA_FINAL.md`
- Resultados da rodada real (60 épocas, com e sem corte por entropia): `hrc-finetune/reports/experimento_2026-07-13_split_unico_epochs60.md`
