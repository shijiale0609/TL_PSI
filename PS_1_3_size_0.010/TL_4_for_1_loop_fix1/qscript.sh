#!/bin/bash
#$ -N P3F1-0.010
#$ -M jshi1@nd.edu
#$ -m abe
#$ -q long
#$ -pe smp 4

conda activate /afs/crc.nd.edu/user/j/jshi1/anaconda3

python NN_training.py 
