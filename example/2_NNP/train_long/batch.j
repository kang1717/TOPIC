#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="composition-NNP-long"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

python3 run.py
