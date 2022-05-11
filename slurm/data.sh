#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
python3 -u /home/djl328/arc/src/data.py
