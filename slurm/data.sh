#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv
for i in 1 2
do
    echo "Generating ${i}-line training programs..."
    python3 -u /home/djl328/arc/src/data.py $i training
    echo "Generating ${i}-line test programs..."
    python3 -u /home/djl328/arc/src/data.py $i validation
done
