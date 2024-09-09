#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4          # Partition name (skylake)
##
#SBATCH --job-name="MD-"
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)



topic_auto_md -np $SLURM_NTASKS total.yaml
