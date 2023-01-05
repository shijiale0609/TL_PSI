#!/bin/bash
#$ -N DL-0.01-loop
#$ -M jshi1@nd.edu
#$ -m abe
#$ -q long@@whitmer
#$ -pe smp 24

conda activate /afs/crc.nd.edu/user/j/jshi1/anaconda3

python NN_training.py 
