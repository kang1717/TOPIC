#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # Cores per node
#SBATCH --partition=loki4       # Partition name (cascade2)
##
#SBATCH --job-name="composition_TOPIC"
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### set path of input.yaml ###
inputfile='input.yaml'
##############################

mpirun -np $SLURM_NTASKS topic_csp $inputfile 
mpirun -np $SLURM_NTASKS topic_post $inputfile
