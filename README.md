# aicup-heart-valve-detection

AI CUP 2025 Fall 

TEAM_8369 – YOLOv12-based Heart Valve Object Detection

Private Leaderboard: 0.964605 (Rank 35)

## Our Environment
GPU: NVIDIA GeForce RTX 3060

CUDA: 12.8

Python: Python 3.10

Note:

All absolute paths (e.g., /NFS/celine/aicup/yoloNew) correspond to the author's training environment 
on a university HPC server. Users should modify ROOT_DIR and dataset paths according to their own 
directory structure before running training scripts.

## Repository Overview
- `S1_data_for_yolo.ipynb`: Prepare raw AI CUP images/labels into YOLO-format folders and text files.
- `S2_training_yolo.ipynb`: Single-fold YOLO training and inference utilities.
- `kfoldS0_prepare.ipynb`, `kfold_5/`: Scripts and data files used to create 5-fold splits (`fold*/fold*_train.txt`, `fold*/fold*_val.txt`) and YOLO dataset YAMLs.
- `kfoldS1_train.ipynb`: Train YOLOv12 models across all folds using the prepared splits.
- `kfoldS4_ensemble_METHOD_infer.ipynb`: Ensemble multiple fold checkpoints for improved validation and test performance.
- `lr_finder.ipynb`, `lrscanS_curve.png`, `lrscanS_summary.csv`: Learning-rate scan experiments.
- `runs/`: Example training outputs (`runs/detect`) and fold checkpoints (`runs/kfold_train`).
- `data9010.yaml`: Base dataset configuration pointing to the author’s HPC paths (update to your own paths before training).


## Dataset Preparation
The official AI CUP dataset includes:
```python
training_image/
training_label/
testing_image/

```
YOLO requires the following structure:
```
dataset/
│── images/
│── labels/
```
Use `S1_data_for_yolo.ipynb` to copy and rename files into the YOLO directory structure and to create train/validation/test splits. Update `data9010.yaml` so the `path` points to the `dataset/` root on your machine.

## K-fold Training and Ensembling
1. Run `kfoldS0_prepare.ipynb` to generate folds.
2. Train each fold with `kfoldS1_train.ipynb`.
3. Review metrics in `kfoldS2_summary.ipynb` and ensemble predictions with `kfoldS4_ensemble_METHOD_infer.ipynb` (leverages `ensemble-boxes`).

