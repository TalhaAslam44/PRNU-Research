## Main results

|                              | ('Unseen cameras (5-fold, 42,083 clips)', 'F1')   | ('Unseen cameras (5-fold, 42,083 clips)', 'AUC')   |   ('Unseen cameras (5-fold, 42,083 clips)', 'EER') |   ('Unseen cameras (5-fold, 42,083 clips)', 'Genuine acc.') | ('Unseen cameras (5-fold, 42,083 clips)', 'Attacks caught')   | ('Unseen cameras (5-fold, 42,083 clips)', 'AI caught')   | ('Unseen datasets (4,716 clips)', 'F1')   | ('Unseen datasets (4,716 clips)', 'AUC')   |
|:-----------------------------|:--------------------------------------------------|:---------------------------------------------------|---------------------------------------------------:|------------------------------------------------------------:|:--------------------------------------------------------------|:---------------------------------------------------------|:------------------------------------------|:-------------------------------------------|
| PCE rule (conventional)      | --                                                | 0.552                                              |                                              0.449 |                                                        22.5 | --                                                            | --                                                       | --                                        | 0.424                                      |
| DINOv2 + linear [frozen]     | 0.680$^{*}$                                       | 0.754$^{*}$                                        |                                              0.324 |                                                        69.5 | 43.3                                                          | 95.0                                                     | 0.824$^{*}$                               | 0.967$^{*}$                                |
| UnivFD-style (CLIP ViT-B/16) | 0.726$^{*}$                                       | 0.804$^{*}$                                        |                                              0.283 |                                                        63.4 | 61.5                                                          | 93.8                                                     | 0.790$^{*}$                               | 0.935$^{*}$                                |
| EfficientNet-B4 (end-to-end) | 0.818$^{*}$                                       | 0.903$^{*}$                                        |                                              0.198 |                                                        76.5 | 71.2                                                          | 98.7                                                     | 0.906                                     | 0.983                                      |
| HVPF (ours)                  | 0.842                                             | 0.921                                              |                                              0.175 |                                                        78.9 | 75.2                                                          | 98.8                                                     | 0.920                                     | 0.987                                      |

## Spoofing

|                              |   AI+PRNU seen gen. |   AI+PRNU unseen gen. |   Camera inj. a=1 |   Camera inj. a=0.5 |
|:-----------------------------|--------------------:|----------------------:|------------------:|--------------------:|
| PCE rule (conventional)      |                95.8 |                  95.8 |              98.5 |                76.2 |
| DINOv2 + linear [frozen]     |                 1.2 |                   1.7 |              48.2 |                57.5 |
| UnivFD-style (CLIP ViT-B/16) |                 0.9 |                   0.8 |              15   |                22.5 |
| EfficientNet-B4 (end-to-end) |                 0.1 |                   0.3 |               3.4 |                 9.4 |
| HVPF (ours)                  |                 0   |                   0   |               0.3 |                 4.8 |

## Adaptive attack

|                             |   Clean F1 cameras |   Clean F1 datasets |   Inj. non-adaptive |   Inj. adaptive |   Usable forgery |   Rem. non-adaptive |   Rem. adaptive | 3-zone inj. / rem.   |
|:----------------------------|-------------------:|--------------------:|--------------------:|----------------:|-----------------:|--------------------:|----------------:|:---------------------|
| HVPF                        |              0.834 |               0.922 |                 1.8 |            67.3 |             49.1 |                 1.8 |            87.3 | 30.9 / 60.0          |
| HVPF-R (ours, robust)       |              0.811 |               0.9   |                 0   |            21.8 |              3.6 |                 1.8 |            49.1 | 1.8 / 20.0           |
| HVPF-R + round 2 (not kept) |              0.792 |               0.929 |                 1.8 |            30.9 |             18.2 |                 1.8 |            47.3 | 7.3 / 23.6           |

## Ablation: features

|                              |   XGBoost F1 |   AUC |
|:-----------------------------|-------------:|------:|
| all features                 |        0.758 | 0.861 |
| without Noiseprint++         |        0.71  | 0.814 |
| without PRNU                 |        0.739 | 0.843 |
| without temporal consistency |        0.74  | 0.84  |
| without residual stats       |        0.751 | 0.858 |
| without FFT                  |        0.756 | 0.859 |

## Ablation: components

|                                                                 |   F1 cameras |   F1 datasets |
|:----------------------------------------------------------------|-------------:|--------------:|
| hybrid stacking (PRNU, Noiseprint++, residual, FFT, TSNCS, ViT) |        0.827 |         0.902 |
| only + reliability-aware fusion                                 |        0.829 |         0.912 |
| only + processing-matched training                              |        0.821 |         0.879 |
| only + camera-calibrated PRNU                                   |        0.836 |         0.925 |
| full HVPF (all three)                                           |        0.842 |         0.92  |

## Three-zone decision

|                           |   accepted |   inconclusive |   accused / flagged |
|:--------------------------|-----------:|---------------:|--------------------:|
| genuine (unseen cameras)  |       54.7 |           38.2 |                 7.2 |
| attacked (unseen cameras) |        6.2 |           36.9 |                56.8 |
| AI (unseen cameras)       |        0.1 |            0.7 |                99.2 |
| genuine (unseen datasets) |       73.8 |           19.3 |                 6.9 |
| AI (unseen datasets)      |        0   |            6.8 |                93.1 |
