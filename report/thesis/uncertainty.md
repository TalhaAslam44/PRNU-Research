# Thesis - validation spread and uncertainty of the final HVPF

From saved predictions (no re-training). 95% camera-level bootstrap intervals in brackets.

## A. Per-fold results (unseen cameras)

| fold   |   clips |   cameras |   val macro F1 |   macro F1 |   AUC genuine vs not |   genuine accepted % |   attacks caught % |   AI caught % |
|:-------|--------:|----------:|---------------:|-----------:|---------------------:|---------------------:|-------------------:|--------------:|
| fold0  |    9800 |        19 |          0.857 |      0.843 |                0.919 |               82.771 |             72.344 |        98.606 |
| fold1  |    8903 |        17 |          0.849 |      0.849 |                0.926 |               79.021 |             77.111 |        98.998 |
| fold2  |    8596 |        16 |          0.852 |      0.825 |                0.903 |               74.264 |             75.218 |        98.762 |
| fold3  |    7253 |        24 |          0.858 |      0.845 |                0.933 |               82.989 |             72.368 |        98.711 |
| fold4  |    7531 |        14 |          0.856 |      0.847 |                0.93  |               75.22  |             79.866 |        99.043 |

|      |   macro F1 |   AUC genuine vs not |   genuine accepted % |   attacks caught % |   AI caught % |
|:-----|-----------:|---------------------:|---------------------:|-------------------:|--------------:|
| mean |      0.842 |                0.922 |               78.853 |             75.381 |        98.824 |
| std  |      0.01  |                0.012 |                4.085 |              3.218 |         0.189 |
| min  |      0.825 |                0.903 |               74.264 |             72.344 |        98.606 |
| max  |      0.849 |                0.933 |               82.989 |             79.866 |        99.043 |

Validation F1 is optimistic: the fusion layer is fitted on the validation fold. Test folds are untouched.

## B. Calibration of the predicted probabilities

**unseen cameras (5-fold, 42,083 clips)** - HVPF ECE 0.009 [0.007, 0.018], Brier 0.226 [0.215, 0.241]

| model        |   accuracy |   ECE |   Brier |   log-loss |
|:-------------|-----------:|------:|--------:|-----------:|
| SVM          |      0.748 | 0.08  |   0.373 |      0.627 |
| XGBoost      |      0.769 | 0.035 |   0.298 |      0.463 |
| GRU          |      0.766 | 0.035 |   0.303 |      0.524 |
| CNN          |      0.768 | 0.034 |   0.304 |      0.536 |
| Transformer  |      0.767 | 0.036 |   0.301 |      0.544 |
| ViT          |      0.781 | 0.142 |   0.322 |      0.56  |
| HVPF (final) |      0.824 | 0.009 |   0.226 |      0.359 |

**unseen datasets (4,716 clips)** - HVPF ECE 0.038 [0.026, 0.052], Brier 0.160 [0.136, 0.187]

| model        |   accuracy |   ECE |   Brier |   log-loss |
|:-------------|-----------:|------:|--------:|-----------:|
| SVM          |      0.759 | 0.051 |   0.402 |      0.737 |
| XGBoost      |      0.757 | 0.03  |   0.348 |      0.666 |
| GRU          |      0.758 | 0.081 |   0.383 |      0.853 |
| CNN          |      0.769 | 0.058 |   0.363 |      0.814 |
| Transformer  |      0.748 | 0.057 |   0.374 |      0.776 |
| ViT          |      0.856 | 0.159 |   0.256 |      0.482 |
| HVPF (final) |      0.889 | 0.038 |   0.16  |      0.308 |

## C. Per-clip uncertainty: does it point at the errors?

AUROC of each uncertainty measure for separating HVPF's wrong decisions from its right ones; error rate when the most uncertain clips are referred to an examiner (coverage = share decided automatically).

**unseen cameras (5-fold, 42,083 clips)**

| measure                   | AUROC errors         |   AURC |   error @ 100% |   error @ 90% |   error @ 80% |   error @ 70% |   error @ 60% |   error @ 50% |
|:--------------------------|:---------------------|-------:|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1 - confidence            | 0.841 [0.822, 0.854] | 0.0501 |           17.6 |          14.3 |          10.8 |           7   |           3.7 |           1.6 |
| entropy                   | 0.836 [0.817, 0.849] | 0.051  |           17.6 |          14.7 |          11.1 |           7.1 |           3.8 |           1.7 |
| base-model disagreement   | 0.794 [0.777, 0.808] | 0.0578 |           17.6 |          15.2 |          12.7 |           9.5 |           7   |           3.2 |
| spread of base p(genuine) | 0.639 [0.623, 0.656] | 0.0989 |           17.6 |          17.2 |          16.4 |          15.5 |          14.3 |          12.6 |

**unseen datasets (4,716 clips)**

| measure                   | AUROC errors         |   AURC |   error @ 100% |   error @ 90% |   error @ 80% |   error @ 70% |   error @ 60% |   error @ 50% |
|:--------------------------|:---------------------|-------:|---------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 1 - confidence            | 0.929 [0.906, 0.949] | 0.0149 |           11.1 |           5.7 |           2.5 |           1.2 |           0.4 |           0.1 |
| entropy                   | 0.919 [0.895, 0.941] | 0.016  |           11.1 |           6.2 |           2.7 |           1.2 |           0.4 |           0.1 |
| base-model disagreement   | 0.831 [0.804, 0.859] | 0.0276 |           11.1 |           8.5 |           6.6 |           4.7 |           1.9 |           0.4 |
| spread of base p(genuine) | 0.733 [0.699, 0.766] | 0.0423 |           11.1 |          10.6 |           8.8 |           7   |           5.4 |           3.4 |

## D. Uncertainty and errors by clip type

**unseen cameras (5-fold, 42,083 clips)**

| type                                    |   clips |   error_pct |   mean_entropy |   mean_confidence |   models_disagreeing_pct |
|:----------------------------------------|--------:|------------:|---------------:|------------------:|-------------------------:|
| genuine: re-encoded + platform-like     |    5568 |      33.567 |          0.571 |             0.657 |                   25.236 |
| attacked: + platform-like               |    7424 |      43.319 |          0.558 |             0.663 |                   25.494 |
| genuine: re-encoded                     |    3712 |      26.67  |          0.489 |             0.737 |                   25.13  |
| genuine: real WhatsApp / YouTube copy   |     834 |      15.827 |          0.474 |             0.741 |                   23.581 |
| attacked: removal / denoise / re-encode |    6496 |      11.345 |          0.279 |             0.865 |                   19.964 |
| attacked: PRNU injection                |    2784 |       6.573 |          0.253 |             0.884 |                   15.817 |
| genuine: native                         |    4805 |       3.163 |          0.132 |             0.953 |                    7.655 |
| AI: + platform-like                     |    4184 |       1.697 |          0.047 |             0.981 |                    3.756 |
| AI: plain                               |    2092 |       1.099 |          0.034 |             0.986 |                    4.135 |
| AI: + injected PRNU                     |    4184 |       0.693 |          0.019 |             0.993 |                    4.84  |

**unseen datasets (4,716 clips)**

| type                |   clips |   error_pct |   mean_entropy |   mean_confidence |   models_disagreeing_pct |
|:--------------------|--------:|------------:|---------------:|------------------:|-------------------------:|
| AI: + platform-like |    1200 |      23.083 |          0.43  |             0.791 |                   26.681 |
| genuine: native     |    1716 |      12.762 |          0.372 |             0.833 |                   21.134 |
| AI: plain           |     600 |       1.333 |          0.093 |             0.965 |                   13.889 |
| AI: + injected PRNU |    1200 |       1.583 |          0.091 |             0.964 |                   11.264 |

