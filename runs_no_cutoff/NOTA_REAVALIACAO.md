# Nota sobre runs_no_cutoff/

Os checkpoints aqui (`V1/seed*/checkpoint.pth`, `V2_dim7/seed*/checkpoint.pth`,
`V2_dim10/seed*/checkpoint.pth`) são cópias exatas dos mesmos checkpoints em
`runs/` (mesmo treino, mesmas sementes) — **não houve novo treino**. Este
diretório existe só para reavaliar esses checkpoints já treinados sem o
corte por entropia (`argmax` cru dos logits), gerando `metrics.json`/
`eval_metrics.json` comparáveis ao layout esperado por
`reports/scripts/aggregate_results.py`.

Na época, essa reavaliação exigiu um checkout do branch `sem-corte-entropia`
(já removido) e um script solto fora do repositório para reaproveitar os
checkpoints sem retreinar. Hoje o mesmo resultado se reproduz num único
branch com `--eval-only`:

```bash
python train_finetune.py --eval-only --variant V1 --context-dim 0 \
  --dataset <dataset_dim0.json> \
  --init-checkpoint-per-seed runs/V1/seed0/checkpoint.pth \
                              runs/V1/seed1/checkpoint.pth \
                              runs/V1/seed2/checkpoint.pth \
  --seeds 0 1 2 --restrict no --out-dir runs_no_cutoff/V1
```

Ver `reports/experimento_2026-07-13_split_unico_epochs60.md` para a
comparação completa com/sem corte.
