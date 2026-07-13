# Tabela ablativa V0/V1/V2

| Variante | Sementes | Top-1 (media +/- dp) | ECE (media +/- dp) | no_action (top-1) | get_connectors (top-1) | get_screws (top-1) | get_wheels (top-1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 (baseline, sem ajuste) | 1 | 0.209 | 0.738 | 0.253 | 0.000 | 0.000 | 0.587 |
| V1 (fine-tune, context_dim=0) | 3 | 0.610 +/- 0.000 | 0.289 +/- 0.000 | 0.781 | 0.367 | 0.082 | 0.000 |
| V2 (contexto 7D) | 3 | 0.659 +/- 0.005 | 0.298 +/- 0.004 | 0.733 | 0.306 | 0.492 | 0.747 |
| V2 (contexto 10D, ablacao) | 3 | 0.663 +/- 0.003 | 0.288 +/- 0.002 | 0.730 | 0.281 | 0.614 | 0.560 |

**Delta V2(7D) - V1 (claim central do artigo): +0.049**
