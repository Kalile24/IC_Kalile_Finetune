# Nota sobre runs_no_cutoff/

Os checkpoints aqui (`V1/seed*/checkpoint.pth`, `V2_dim7/seed*/checkpoint.pth`,
`V2_dim10/seed*/checkpoint.pth`) são cópias exatas dos mesmos checkpoints em
`runs/` (mesmo treino, mesmas sementes) — **não houve novo treino**. Este
diretório existe só para reavaliar esses checkpoints já treinados com
`evaluate()` sem o corte por entropia (`argmax` cru dos logits, código do
branch `sem-corte-entropia`), gerando `metrics.json`/`eval_metrics.json`
comparáveis ao layout esperado por `reports/scripts/aggregate_results.py`.

Ver `reports/experimento_2026-07-13_split_unico_epochs60.md` para a
comparação completa com/sem corte.
