#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
python3 -u /home/djl328/arc/src/transformer.py \
	--train \
	--name '10k-2r2z-2048' \
	--training-data /home/djl328/arc/data/10000-2r2z-train.exs \
	--validation-data /home/djl328/arc/data/10000-2r2z-test.exs \
	--model-dim 2048 \
	--learning-rate 0.00005 \
	-n 5 \
	--epochs 1000000 \
	--batch-size 64 \
	--sample-freq 10 \
	--log-freq 3
