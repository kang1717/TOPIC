from module_structure import *
from module_lammps import *
from module_scrutinize import *
from module_log import *
import time
from time import time
import mpi4py
from mpi4py import MPI
import os
import shutil

########### input #############
spg_tolerance =[ [[1,1], 3.3*0.8],
                 [[1,2], 3.1*0.8],
                 [[2,2], 2.8*0.8]]
composition = [1 for i in range(16)] + [2 for i in range(8)]
volume      = 1100.0
elements = 'Ta P'
###############################


# MPI setting
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
corenum = comm.Get_size()


if os.path.isdir(str(rank)) == False:
  shutil.copytree('example',str(rank))

os.chdir(str(rank)) 


success_1   = 0
iteration_1 = 0

#while success_1 == 0 and iteration_1 < 100:

write_log_head()

while 1:
 
  E0 =  1000
  E  =  1000
  V  =  100
  iteration_1 += 1
  t1 = time() 
  fail_1=1
  while fail_1==1: 
    # generate POSCAR file
    fail_1 = 0
    try:
      spg = randspg(composition,elements,volume,spg_tolerance)
    except:
      fail_1=1

  fail_2   = 24
  iteration_2 = 0

  t2 = time()

  T0  = t2-t1

  ########################## LJ & spring model #############################
  while fail_2 > 0 and iteration_2 < 1:
 
 
    cation_dict = make_oxygen()

    # write lammps input: in.all
    lammps_write('POSCAR',cation_dict)

    # convert POSCAR to coo 
    POSCAR2cooall('POSCAR')

    # run lammps
    E0,V0 = run_lammps('in.all')

    # convert coo_out to CONTCAR 
    coo2CONTCAR('coo_out')

    # scrutinize
    fail_2 = scrutinize('CONTCAR',cation_dict)

    if (E0 == 0 and V0 == 0) == False:
      iteration_2 += 1

    else:
      continue

  t3 = time()
  T1 = t3-t2

  ############################ short NNP ##############################
  if fail_2 == 0:

    # convert POSCAR to coo
    POSCAR2coo("CONTCAR",'coo_out2')

    # run lammps
    E,V = run_lammps('in.nnp')

    # convert coo_out to CONTCAR 
    coo2CONTCAR('coo_nnp','CONTCAR2')

    # scrutinize
    fail_3 = scrutinize('CONTCAR2',cation_dict)

  else:
    fail_3 = 100

  t4 = time()
  T2 = t4-t3

  if fail_3 == 0:

    # convert POSCAR to coo
    POSCAR2coo("CONTCAR2",'coo_out3')

    # run lammps
    E,V = run_lammps('in.nnplong')

    # convert coo_out to CONTCAR 
    coo2CONTCAR('coo_nnp2','CONTCAR3')

    # scrutinize
    fail_4 = scrutinize('CONTCAR3',cation_dict)

  else:
    fail_4 = 100

  t5 = time()
  T3 = t5-t4
  write_log(rank,spg,iteration_1,iteration_2,T0,T1,T2,T3,fail_2,fail_3, E, V)

