#!/bin/sh

echo "Running randomized ID and compute coefficient by solving Least-square problem"
python rID.py -m 1 --random

echo "Running randomized ID and compute coefficient by solving Least-square problem"
python rID.py -m 2 --random

echo "Running randomized ID and updating coefficient based on approximation residual of old basis"
python rID.py -m 3 --random

echo "Running randomized ID and updating coefficient by updating QR of basis"
python rID.py -m 4 --random

echo "Running ID without sketching and compute coefficient by solving Least-square problem"
python rID.py -m 1 --no-random

# #set variables
# i=1
# word="dog"
# #read in template one line at the time, and replace variables
# #(more natural (and efficient) way, thanks to Jonathan Leffler)
# while read line
# do
#     eval echo "$line"
# done < "./template.txt"