#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
python3 -u /home/djl328/arc/src/transformer.py \
	--train \
	--name 'simple' \
	--data /home/djl328/arc/data/10k-simple-exs.dat \
	-n 1 \
	--epochs 1000000 \
	--batch-size 32 \
	--sample-freq 10 \
	--log-freq 1
