#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
python3 -u /home/djl328/arc/src/transformer.py train /home/djl328/arc/data/10k-zless-exs.dat 1
