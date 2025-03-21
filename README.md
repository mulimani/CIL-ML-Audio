# Class-Incremental Learning for Multi-Label Audio Classification

This repo contains the code for our papers: [**Class-Incremental Learning for Multi-Label Audio Classification**](https://ieeexplore.ieee.org/abstract/document/10447952) and 
[**A Closer Look at Class-Incremental Learning for Multi-Label Audio Classification**](https://doi.org/10.1109/TASLPRO.2025.3547233).
Experiments are performed on a dataset (AudioSet and FSD50K) with 50 sound classes, with an initial classification task containing 30 base classes and 4 incremental phases of 5 classes each. After each phase, the system is tested for multi-label classification with the entire set of classes learned so far. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data 

### AudioSet

* We selected [50 major classes](data/AudioSet_classes.txt) from temporally strong [AudioSet](https://research.google.com/audioset/download_strong.html).
* The file [Audioset_sounds.txt](data/Audioset_sounds.txt) describes the number of occurrences of
 each class across the 95645 excerpts of length 10s from the development set and the 15289 excerpts of length 10s from the evaluation set.

### FSD50K

* We selected [50 major classes](data/FSD50K_classes.txt) from [FSD50K](https://zenodo.org/records/4060432).
* The file [FSD50K_sounds.txt](data/FSD50K_sounds.txt) describes the number of occurrences of
 each class across the 284447 excerpts of length 1s from the development set and the 97798 excerpts of length 1s from the evaluation set.

## Data preparation for incremental learning

We created text files for training and evaluation in each phase and stored in data_txt folder. For example:

Prepare the text files includes path for the audio files and corresponding labels present in each phase as follows (and stored in data_txt folder):
* train_phase_0.txt (30 classes)
```
/scratch/project_2003370/alex/Audioset/train/Yb0RFKhbpFJA_30000.wav_[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0.]
/scratch/project_2003370/alex/Audioset/train/YNQNTnl0zaqU_70000.wav_[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0.]
 ....
```
* test_phase_0.txt (30 classes)
```
/scratch/project_2003370/alex/Audioset/eval/Ys9d-2nhuJCQ_30000.wav_[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 1. 0.]
/scratch/project_2003370/alex/Audioset/eval/YYxlGt805lTA_30000.wav_[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0.]
 ....
```
* train_phase_1.txt (5 classes)
```
/scratch/project_2003370/alex/Audioset/train/YcUAe_N9oODs_0.wav_[0. 0. 0. 1. 0.]
/scratch/project_2003370/alex/Audioset/train/Y3rmbKeJ_ydE_30000.wav_[0. 1. 0. 0. 0.]
 ....
```
* test_phase_0_1.txt (35 classes)
```
/scratch/project_2003370/alex/Audioset/eval/YQhFtABqY9cs_220000.wav_[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
/scratch/project_2003370/alex/Audioset/eval/YrZfSTca9wCw_210000.wav_[0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 ....
```
* train_phase_2.txt (5 classes)
```
/scratch/project_2003370/alex/Audioset/train/Y4PPmyY_-YrA_30000.wav_[0. 0. 0. 0. 1.].npy
/scratch/project_2003370/alex/Audioset/train/YO35jXasNYxc_30000.wav_[0. 1. 0. 0. 0.].npy
 ....
```
* test_phase_0_1_2.txt (40 classes)
```
/scratch/project_2003370/alex/Audioset/eval/YKNdwhLlOv0I_30000.wav_[0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
/scratch/project_2003370/alex/Audioset/eval/YoczCxi-5PHQ_0.wav_[0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 ....
```
* train_phase_3.txt (5 classes)
```
/scratch/project_2003370/alex/Audioset/train/YVPueASe6yxY_440000.wav_[1. 0. 0. 0. 0.].npy
/scratch/project_2003370/alex/Audioset/train/Yk-VZj_uYyjg_520000.wav_[0. 0. 0. 1. 0.].npy
 ....
```
* test_phase_0_1_2_3.txt (45 classes)
```
/scratch/project_2003370/alex/Audioset/eval/YN_LKZjw9DLk_60000.wav_[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
/scratch/project_2003370/alex/Audioset/eval/YN4PeCoQzfBo_70000.wav_[0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 ....
```
* train_phase_4.txt (5 classes)
```
/scratch/project_2003370/alex/Audioset/train/YVPueASe6yxY_440000.wav_[1. 0. 0. 0. 0.].npy
/scratch/project_2003370/alex/Audioset/train/Yk-VZj_uYyjg_520000.wav_[0. 0. 0. 1. 0.].npy
 ....
```
* test_phase_0_1_2_3_4.txt (50 classes)
```
/scratch/project_2003370/alex/Audioset/eval/YR-d-5b54PIk_28000.wav_[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0.]
/scratch/project_2003370/alex/Audioset/eval/Y2Fbt9QiLWWc_370000.wav_[0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 ....
```
## Normalization layers
Select the models with different normalization layers listed below from [models](models) folder and include in [main_audioset.py](main_audioset.py), default one is BN layer.
* [cnn14_pann_lin.py](models/cnn14_pann_lin.py) for Batch Normalization (BN) layer
* [cnn14_pann_lin_CN.py](models/cnn14_pann_lin_CN.py) for Continual Normalization (CN) layer
* [cnn14_pann_lin_GN.py](models/cnn14_pann_lin_GN.py) for Group Normalization (GN) layer


## Training and Evaluation

Following scripts contain both training and evaluation codes.

### CIL of sounds without exemplars

```
bash ./scripts/run_audioset.sh
bash ./scripts/run_audioset_with_cka.sh
```

## Acknowledgement

This repo is based on aspects of [Essentials for Class Incremental Learning](https://github.com/sud0301/essentials_for_CIL/tree/main)

## Citation

```BibTeX

@inproceedings{10447952,
  author={Mulimani, Manjunath and Mesaros, Annamaria},
  title={Class-Incremental Learning for Multi-Label Audio Classification}, 
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},     
  pages={916-920},  
  year={2024}
}


```
```BibTeX

@article{10909318,
  author={Mulimani, Manjunath and Mesaros, Annamaria},  
  title={A Closer Look at Class-Incremental Learning for Multi-Label Audio Classification},  
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  volume={33},
  pages={1293-1306},  
  year={2025}
}
```