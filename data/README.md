# Dataset

The dataset was merged from ten single-class public datasets. Images are **not** redistributed in this repository; download them from the original pages and convert the labels to YOLO format with the class mapping below.

| ID | Class | Source | Images on source page | Licence |
|---:|---|---|---:|---|
| 0 | mask | Maranhão, A. *Face Mask Detection*, Kaggle – https://www.kaggle.com/datasets/andrewmvd/face-mask-detection | 853 | CC0 1.0 |
| 1 | helmet | Maranhão, A. *Safety Helmet Detection*, Kaggle – https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection | 5000 | see source page |
| 2 | belt (harness) | Treino, *safety-belt-detection*, Roboflow Universe – https://universe.roboflow.com/treino-rwdcq/safety-belt-detection | 3862 | see source page |
| 3 | vest | Echo, *Safety Vests*, Roboflow Universe – https://universe.roboflow.com/echo-dotf3/safety-vests-c89yh (Kaggle copy: https://www.kaggle.com/datasets/adilshamim8/safety-vests-detection-dataset) | 3897 | CC BY 4.0 |
| 4 | glasses | test, *Goggles images*, Roboflow Universe – https://universe.roboflow.com/test-cysdu/goggles-images-dmcbm-znuk8 | 795 | CC BY 4.0 |
| 5 | welding_helmet | UTP, *Welding Helmet*, Roboflow Universe – https://universe.roboflow.com/utp-jtbn5/welding-helmet | 1098 | see source page |
| 6 | ear_protection | labellingapd, *ear protection*, Roboflow Universe – https://universe.roboflow.com/labellingapd/ear-protection | 70 | CC BY 4.0 |
| 7 | toolbox | AvakaI, *toolbox*, Roboflow Universe – https://universe.roboflow.com/avakai/toolbox-jdlpp | 135 | see source page |
| 8 | safety_gloves | labellingapd, *safety gloves*, Roboflow Universe – https://universe.roboflow.com/labellingapd/safety-gloves-i0xhe | 70 | see source page |
| 9 | protective_suit | Hassnae, *yellow chemical protective suit*, Roboflow Universe – https://universe.roboflow.com/hassnae/yellow-chemical-protective-suit | 182 | see source page |

## Composition used in the paper

* **Version 1 (five classes, `ISG_Final_Dataset`)** – at most 1000 images per class (mask 853, glasses 442), random 80/20 split: 3436 training and 859 validation images.
* **Version 2 (ten classes, `ISG_Final_Dataset2`)** – the five classes of version 1 plus the five Roboflow sources: 4969 training images (138 without objects, 13 166 boxes) and 516 validation images (20 without objects, 1236 boxes). The validation set was built with `training/v2_ten_classes/build_validation_set.py`.

## Notes for re-use

* One training image of the vest class has a very long file name; when the archive is extracted on Windows, the full path can exceed the 260-character limit and the file is silently skipped (4968 instead of 4969 training images). Extract the archive into a short path or enable long paths.

* Roboflow exports contain several augmented copies of a source image whose file names share the part before `.rf.`; split the data **by this original file name** so that all copies stay on the same side.
* In the face-mask source, only the `with_mask` boxes should be mapped to class 0; the paper's models used all three categories (see the paper, Section 4.7).
* Each source is annotated for one class only; images may contain unlabelled items of other classes.

`data.yaml` is a template for the ten-class dataset. `sources.csv` lists the same sources in machine-readable form together with the per-class image and object counts of the ten-class dataset (training image counts per class are known only for the five classes of version 1).
