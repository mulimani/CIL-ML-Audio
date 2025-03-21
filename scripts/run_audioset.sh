#!/bin/bash

python main_audioset.py --new-classes 5 --start-classes 30 --cosine --kd --w-kd 1 \
       --lr 0.01 --lr-ft 0.001 --num-workers 20 --batch-size 32 --epochs 120 --save