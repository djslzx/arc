#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
# python3 -u /home/djl328/arc/src/transformer.py  /home/djl328/arc/models/tiny/tf_model_1643860942_107930.pt /home/djl328/arc/data/tiny-exs.dat
for model in /home/djl328/arc/models/tiny/*
do
    python3 -u /home/djl328/arc/src/transformer.py  $model /home/djl328/arc/data/tiny-exs.dat
done
