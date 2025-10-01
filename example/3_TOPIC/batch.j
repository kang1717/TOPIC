#!/bin/bash

inputfile='input.yaml'
NUM_CORE=DEFINE_NUM_CORE

mpirun -np $NUM_CORE topic_csp $inputfile 
mpirun -np $NUM_CORE topic_post $inputfile
