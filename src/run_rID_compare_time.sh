#!/bin/sh

echo "To compare the computation time, we first get a sketch of the original matrix (size 400*1000) and then run ID without randomized sketching"

echo "Running ID and compute coefficient by solving Least-square problem"
python rID.py -m 1 --no-random

echo "Running ID and compute coefficient by  X = inv(A_Omega^T A_Omega) * A_Omega^T A"
python rID.py -m 2 --no-random

echo "Running ID and updating coefficient based on approximation residual of old basis"
python rID.py -m 3 --no-random

echo "Running ID and updating coefficient by updating QR of basis"
python rID.py -m 4 --no-random
