# **A Closer Look at Class-Incremental Learning for Multi-label Audio Classification**


This repo contains the code for our work 'A Closer Look at Class-Incremental Learning for Multi-label Audio Classification' written using PyTorch..

## Data 

### AudioSet

* We selected [50 major classes](AudioSet_classes.txt) from temporally strong [AudioSet](https://research.google.com/audioset/download_strong.html).
* The file [Audioset_sounds.txt](Audioset_sounds.txt) describes the number of occurrences of
 each class across the 95645 excerpts of length 10s from the development set and the 15289 excerpts of length 10s from the evaluation set.

### FSD50K

* We selected [50 major classes](FSD50K_classes.txt) from [FSD50K](https://zenodo.org/records/4060432).
* The file [FSD50K_sounds.txt](FSD50K_sounds.txt) describes the number of occurrences of
 each class across the 284447 excerpts of length 1s from the development set and the 97798 excerpts of length 1s from the evaluation set.

## Models

* CNN14
* Vgg-like

## Methods

* CIL of sounds without exemplars
* CIL of sounds with exemplars



Code will be made available soon. Stay tuned!