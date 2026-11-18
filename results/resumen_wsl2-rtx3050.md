# Resultados — wsl2-rtx3050

## Tiempo por época (promedio de las épocas 2 y 3)

| etapa_lote | dispositivo | cambio | reps | seg_media | seg_std | img_s | val_acc | speedup_vs_E0 | speedup_vs_anterior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | cpu | Código original del lab (CPU) | 3 | 396.19 | 1.46 | 153.72 | 0.95 | 1.00 | — |
| E1 | gpu | Mismo código con GPU visible | 5 | 55.64 | 0.95 | 1,094.85 | 0.92 | 7.12 | 7.12 |
| E2 | cpu | .npy + tf.data (CPU) | 3 | 304.57 | 4.80 | 199.98 | 0.95 | 1.30 | 0.18 |
| E3 | gpu | .npy + tf.data (GPU) | 5 | 38.78 | 0.46 | 1,570.73 | 0.98 | 10.22 | 7.85 |
| E4-256 | gpu | Lote grande | 5 | 26.45 | 0.29 | 2,302.83 | 0.94 | 14.98 | 1.47 |
| E4-512 | gpu | Lote grande | 5 | 26.20 | 3.79 | 2,382.90 | 0.05 | 15.12 | 1.01 |
| E5 | gpu | Precisión mixta | 5 | 28.53 | 0.38 | 2,135.13 | 0.95 | 13.89 | 0.92 |
| E6 | gpu | XLA (jit_compile) | 5 | 7.85 | 0.02 | 7,762.39 | 0.95 | 50.50 | 3.64 |

## Speedups reportados

| speedup | t_antes_s | t_despues_s | valor |
| --- | --- | --- | --- |
| Fase 1: preparación secuencial / paralela | 54.23 | 9.29 | 5.84× |
| Fase 2, efecto del GPU: E2 / E3 | 304.57 | 38.78 | 7.85× |
| Total: E0 / mejor etapa (E6) | 396.19 | 7.85 | 50.50× |
| Mismo código en GPU: E0 / E1 | 396.19 | 55.64 | 7.12× |

## Transferencia CPU → GPU (un lote)

| etapa_lote | reps | mb_por_lote | ms_media | ms_std | gb_s |
| --- | --- | --- | --- | --- | --- |
| E1 | 5 | 3.146 | 0.768 | 0.088 | 4.096 |
| E3 | 5 | 0.786 | 0.733 | 0.026 | 1.073 |
| E4-256 | 5 | 3.146 | 0.721 | 0.056 | 4.364 |
| E4-512 | 5 | 6.291 | 0.784 | 0.126 | 8.021 |
| E5 | 5 | 3.146 | 0.823 | 0.131 | 3.823 |
| E6 | 5 | 3.146 | 0.779 | 0.148 | 4.040 |

## Fase 1: preparación de datos

| etapa | reps | seg_media | seg_std | img_s | speedup |
| --- | --- | --- | --- | --- | --- |
| secuencial | 3 | 54.23 | 0.20 | 1,363.67 | 1.00 |
| paralelo | 3 | 9.29 | 0.65 | 7,986.57 | 5.84 |

## Diagnóstico (segundos por época equivalente)

| etapa | lote | seg_epoca | img_s |
| --- | --- | --- | --- |
| datos_generator | 64 | 49.55 | 1,229.10 |
| datos_arrays | 64 | 2.18 | 27,970.70 |
| matematica | 64 | 37.38 | 1,629.40 |

## Uso de recursos (media por segundo, sin los primeros 5 s)

| etapa_lote | reps | idle_gpu_util | gpu_util | gpu_util_p95 | idle_vram_mb | vram_max_mb | cpu_total | nucleos_ocupados |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 3 | 1.9 | 2.8 | 10.7 | 761.0 | 770 | 85.0 | 12.0 |
| E1 | 5 | 1.8 | 48.9 | 54.9 | 770.0 | 1,551 | 13.5 | 0.0 |
| E2 | 3 | 3.5 | 2.6 | 10.7 | 762.0 | 762 | 84.5 | 12.0 |
| E3 | 5 | 4.5 | 64.9 | 70.5 | 747.0 | 1,545 | 7.2 | 0.0 |
| E4-256 | 5 | 2.5 | 85.7 | 92.6 | 762.0 | 3,607 | 3.8 | 0.0 |
| E4-512 | 5 | 5.3 | 90.0 | 96.0 | 706.8 | 4,481 | 3.2 | 0.0 |
| E5 | 5 | 1.8 | 75.0 | 87.0 | 718.0 | 1,932 | 4.7 | 0.0 |
| E6 | 5 | 1.0 | 47.7 | 91.0 | 711.0 | 2,881 | 9.3 | 0.0 |
