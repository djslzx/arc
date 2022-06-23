#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate tenv

n_train_programs=50_000
n_train_workers=100
n_test_programs=500
n_test_workers=10

time=$(date -Iseconds)
echo "Starting data gen at time=${time}..."
for ((i=1;i<=20;i++))
do
    echo "START ${i}-LINE PROGRAMS"
    echo "Generating ${n_train_programs} ${i}-line training programs..."
    python3 -u /home/djl328/arc/src/data.py /home/djl328/arc/data/ train 0 0 $i $n_train_programs $n_train_workers $time
    echo "Generating ${n_test_programs} ${i}-line test programs..."
    python3 -u /home/djl328/arc/src/data.py /home/djl328/arc/data/ test 0 0 $i $n_test_programs $n_test_workers $time
    echo "END ${i}-LINE PROGRAMS"
done
