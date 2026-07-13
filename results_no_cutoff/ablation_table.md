# Tabela ablativa V0/V1/V2

| Variante | Sementes | Top-1 (media +/- dp) | ECE (media +/- dp) | no_action (top-1) | get_connectors (top-1) | get_screws (top-1) | get_wheels (top-1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 (baseline, sem ajuste) | 1 | 0.041 | 0.861 | 0.000 | 0.057 | 0.000 | 0.773 |
| V1 (fine-tune, context_dim=0) | 3 | 0.369 +/- 0.001 | 0.337 +/- 0.001 | 0.192 | 0.677 | 0.968 | 0.724 |
| V2 (contexto 7D) | 3 | 0.678 +/- 0.001 | 0.155 +/- 0.000 | 0.570 | 0.882 | 1.000 | 0.987 |
| V2 (contexto 10D, ablacao) | 3 | 0.734 +/- 0.001 | 0.123 +/- 0.001 | 0.642 | 0.928 | 1.000 | 0.987 |

**Delta V2(7D) - V1 (claim central do artigo): +0.309**
