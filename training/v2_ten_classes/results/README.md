# Training outputs of the ten-class model

Files from the Ultralytics run folder of the final training run (`ISG_Egitim_Raporu_20260529_1413`):

| File | Content |
|---|---|
| `args.yaml` | full training configuration |
| `results.csv` | per-epoch training and validation losses, precision, recall and mAP |
| `results.png` | the same curves as an Ultralytics plot |
| `BoxPR_curve.png`, `BoxF1_curve.png` | precision–recall and F1–confidence curves on the validation set |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | confusion matrices on the validation set |
| `labels.jpg` | label statistics of the training set |
| `val_batch_pred.png` | predictions on one validation batch |
| `training_log_parsed.csv` | metrics parsed from the notebook output (same values as `results.csv`) |

Most validation images are copies of training images (431 of 516), so the validation metrics are optimistic; see `evaluation/README.md` for the independent test.
