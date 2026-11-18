# Resultados — wsl2-rtx3050

## Tiempo por época (promedio de las épocas 2 y 3)

| etapa_lote | dispositivo | cambio | reps | seg_media | seg_std | img_s | val_acc | speedup_vs_E0 | speedup_vs_anterior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | cpu | Código original del lab (CPU) | 3 | 396.19 | 1.46 | 153.72 | 0.95 | 1.00 | — |
| E1 | gpu | Mismo código con GPU visible | 5 | 55.64 | 0.95 | 1,094.85 | 0.92 | 7.12 | 7.12 |

## Speedups reportados

| speedup | t_antes_s | t_despues_s | valor |
| --- | --- | --- | --- |
| Fase 1: preparación secuencial / paralela | — | — | pendiente |
| Fase 2, efecto del GPU: E2 / E3 | — | — | pendiente |
| Total: E0 / mejor etapa (E1) | 396.19 | 55.64 | 7.12× |
| Mismo código en GPU: E0 / E1 | 396.19 | 55.64 | 7.12× |

## Transferencia CPU → GPU (un lote)

| etapa_lote | reps | mb_por_lote | ms_media | ms_std | gb_s |
| --- | --- | --- | --- | --- | --- |
| E1 | 5 | 3.146 | 0.768 | 0.088 | 4.096 |

## Uso de recursos (media por segundo, sin los primeros 5 s)

| etapa_lote | reps | idle_gpu_util | gpu_util | gpu_util_p95 | idle_vram_mb | vram_max_mb | cpu_total | nucleos_ocupados |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 3 | 1.9 | 2.8 | 10.7 | 761.0 | 770 | 85.0 | 12.0 |
| E1 | 5 | 1.8 | 48.9 | 54.9 | 770.0 | 1,551 | 13.5 | 0.0 |
