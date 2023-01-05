#!/bin/bash
#$ -N P3F3-0.010
#$ -M jshi1@nd.edu
#$ -m abe
#$ -q long@@whitmer
#$ -pe smp 24

conda activate /afs/crc.nd.edu/user/j/jshi1/anaconda3

python NN_training.py 
