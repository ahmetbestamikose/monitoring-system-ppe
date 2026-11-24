# Evaluation scripts and results

## 1. Ten-class model (version 2): overlap and independent test

`v2_ten_classes/overlap_check_v2.py` compares every validation image of `ISG_Final_Dataset2` with the training images by original file name (the part before the `_jpg.rf.` suffix of Roboflow exports) and by MD5 checksum, and selects an independent test set from the validation folder of the five-class dataset (`ISG_Final_Dataset`), which was not used to build the ten-class training set.

* 431 of the 516 validation images have a copy in the training set, among them **all** validation images of the five added classes (welding helmet, ear protection, toolbox, safety gloves, protective suit).
* 446 images of the five-class validation folder (645 masks, 778 helmets, 48 harnesses, 136 vests, 42 glasses) have no copy in the ten-class training or validation set and form the independent test set (`subsets_v2.json`).

`v2_ten_classes/run_eval_v2.py` evaluates the final weights on the validation set and on the test set (Ultralytics 8.4.216 on CPU; polygon outlines in the labels were replaced by their enclosing boxes). Results (`res_val_all.json`, `res_test_indep.json`):

| Class | Validation images (copies) | Validation mAP@0.5 | Test images (objects) | Test P | Test R | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mask | 51 (19) | 0.951 | 149 (645) | 0.962 | 0.852 | 0.923 | 0.628 |
| Helmet | 64 (31) | 0.918 | 158 (778) | 0.937 | 0.823 | 0.901 | 0.575 |
| Harness (belt) | 61 (59) | 0.993 | 29 (48) | 0.809 | 0.704 | 0.821 | 0.440 |
| Vest | 68 (56) | 0.954 | 72 (136) | 0.925 | 0.787 | 0.865 | 0.613 |
| Glasses | 19 (14) | 0.925 | 38 (42) | 0.944 | 0.802 | 0.834 | 0.456 |
| **Five original classes** | 263 (179) | 0.948 | 446 (1649) | 0.915 | 0.794 | 0.869 | 0.542 |
| Welding helmet / ear protection / toolbox / safety gloves / protective suit | all copies | 0.984 / 0.995 / 0.993 / 0.995 / 0.894 | – | – | – | – | – |

The re-evaluation gives mAP@0.5 = 0.960 on the full validation set (training log: 0.969); the difference comes mainly from the protective suit, several of whose validation labels mix polygon and box rows. `run_curves.py` exports the precision–recall and F1 curves used for the figure in the paper (`fig_pr_curves_v6.py`).

CPU timing with the ten-class weights (`benchmark_cpu_result_v2.json`, two vCPUs, Intel Xeon 2.1 GHz): inference 565 ms, mobile server cycle 565 ms, desktop server cycle with the ear passes 2583 ms. In the same session the five-class weights took 566 ms per prediction; the earlier session below gave 418 ms, so the timings of this shared cloud machine vary between sessions.

## 2. Train/validation overlap (five-class model, version 1)

`overlap_check.py` looks for validation images that have a copy in the training set. Roboflow exports name augmented copies of a photograph `<original>_<ext>.rf.<hash>.<ext>`, so files sharing the part before `.rf.` show the same photograph; exact duplicates are found by MD5.

Result for `ISG_Final_Dataset` (859 validation images): 327 images have a copy in the training set (harness 198/230, vest 102/187, glasses 25/68, mask 2/183 exact duplicates, helmet 0/191); 532 images are clean.

`evaluate_subsets.py` re-evaluates the model on the full, clean and overlapping subsets (`subset_results_v1_5class.json`; values are precision, recall, mAP@0.5, mAP@0.5:0.95; Ultralytics 8.4.216 on CPU):

| Class | Full set mAP@0.5 | Clean subset mAP@0.5 | Overlap subset mAP@0.5 |
|---|---:|---:|---:|
| Harness (belt) | 0.841 | 0.694 | 0.868 |
| Vest | 0.909 | 0.874 | 0.934 |
| Glasses | 0.888 | 0.850 | 0.943 |
| Mask | 0.928 | 0.928 | – |
| Helmet | 0.868 | 0.869 | – |
| All classes | 0.887 | 0.843 | – |

The full-set values differ slightly from the training log (0.897) because a newer Ultralytics version was used; the subsets are compared within the same run. For the ten-class model see section 1.

## 3. CPU timing (earlier session)

`benchmark_cpu.py` times the server pipelines without camera and network (`benchmark_cpu_result.json`), on two vCPUs of an Intel Xeon processor at 2.1 GHz without GPU, using the five-class weights (same network size as the ten-class model):

| Configuration | Mean time per frame |
|---|---:|
| YOLOv8m inference only | 418 ms |
| Mobile server cycle (without the 60 ms pause) | 460 ms |
| Desktop server cycle with ear-protection passes | 2006 ms |
