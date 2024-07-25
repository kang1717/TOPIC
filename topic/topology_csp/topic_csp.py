from topic.topology_csp.module_structure import make_oxygen, coo2CONTCAR, randspg
from topic.topology_csp.module_lammps import lammps_write, POSCAR2cooall, POSCAR2coo, run_lammps, run_lj_lammps
from topic.topology_csp.module_scrutinize import scrutinize
from topic.topology_csp.module_log import write_log_head, write_log
import time
from time import time
import mpi4py
from mpi4py import MPI
import os, sys, shutil, yaml

########### input #############

def main()
    input_file = str(sys.argv[1])
    with open(input_file, 'r') as f:
        total_yaml = yaml.safe_load(f)

    generation      = total_yaml['generation']
    volume          = total_yaml['volume']
    material        = total_yaml['material']
    bond_dict       = total_yaml['distance_constraint']
    factor          = total_yaml['constraint_factor']
    factor          = 1

    # Getting cation information, and sort with number of atoms
    cat_info = {k: v for k, v in material.items() if k not in {'Li', 'O'}}
    cat_info = sorted(cat_info.items(), key=lambda item: item[1], reverse=True)

    # bond distance for metal atoms
    m1_m1 = bond_dict[f"{cat_info[0][0]}-{cat_info[0][0]}"]
    m2_m2 = bond_dict[f"{cat_info[1][0]}-{cat_info[1][0]}"]
    if f"{cat_info[0][0]}-{cat_info[1][0]}" in bond_dict:
        m1_m2 = bond_dict[f"{cat_info[0][0]}-{cat_info[1][0]}"]
    else:
        m1_m2 = bond_dict[f"{cat_info[1][0]}-{cat_info[0][0]}"]

    # Distance constraint for random structure generation
    spg_tolerance = [ [[1,1], m1_m1*factor],
                      [[1,2], m1_m2*factor],
                      [[2,2], m2_m2*factor] ]
    composition = [1 for i in range(cat_info[0][1])] + [2 for i in range(cat_info[1][1])]
    elements = ' '.join([item[0] for item in cat_info])

    ###############################


    # MPI setting
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    corenum = comm.Get_size()

    if os.path.isdir(str(rank)) == False:
        shutil.copytree('input', str(rank))
    os.chdir(str(rank))

    write_log_head()

    success_1   = 0
    iteration_1 = 0
    for i in range(int(generation/corenum)):     # modified
        iteration_1 += 1

        E0 =  1000
        E  =  1000
        V  =  100
        t1 = time()

        # 1. Generate initial POSCAR file
        fail_1 = 1
        while fail_1 == 1:
            try:
                spg = randspg(composition, elements, volume, spg_tolerance)
                fail_1 = 0
            except:
                fail_1 = 1
        t2 = time()

        # 2. LJ & spring model 
        cation_dict = make_oxygen()             # set oxygen inbetween two cations
        lammps_write('POSCAR', cation_dict)     # write lammps input: in.all
        POSCAR2cooall('POSCAR')                 # convert POSCAR to coo 
        shutil.copyfile('POSCAR', f'POSCAR_{i+1}')

        with open('lj_results', 'a') as f:
            E0, V0 = run_lj_lammps('in.all')
            f.write(f"Generation {i+1}, Energy: {E0:.3f}, Volume: {V0:.3f}\n")
            #os.rename('out.xyz', f'out_{i+1}.xyz')

        if E0 != 0 or V0 != 0:
            coo2CONTCAR('coo_out')
            fail_2 = scrutinize('CONTCAR', cation_dict, total_yaml)
        else:
            continue
        t3 = time()

        # 3. short NNP 
        if fail_2 == 0:
            os.rename(f'out.xyz', f'out_{i+1}_lj.xyz')
            POSCAR2coo("CONTCAR", 'coo_out2')               # convert POSCAR to coo
            shutil.copyfile('CONTCAR', f'CONTCAR_lj_{i+1}')

            E,V = run_lammps('in.nnp')
            os.rename('out.xyz', f'out_{i+1}_nnp.xyz')

            coo2CONTCAR('coo_nnp', 'CONTCAR2')              # convert coo_out to CONTCAR 
            shutil.copyfile('CONTCAR2', f'CONTCAR_nnp_short_{i+1}')

            fail_3 = scrutinize('CONTCAR2', cation_dict, total_yaml)
        else:
            fail_3 = 100
        t4 = time()

        # 3. long NNP 
        if fail_3 == 0:
            POSCAR2coo("CONTCAR2", 'coo_out3')              # convert POSCAR to coo

            E,V = run_lammps('in.nnplong')

            coo2CONTCAR('coo_nnp2', 'CONTCAR3')             # convert coo_out to CONTCAR 
            shutil.copyfile('CONTCAR3', f'CONTCAR_nnp_long_{i+1}')

            fail_4 = scrutinize('CONTCAR3', cation_dict, total_yaml)
        else:
            fail_4 = 100

        if fail_4 == 0:
            shutil.copyfile('CONTCAR3', f'CONTCAR_success_{i+1}')
        t5 = time()


        T0 = t2-t1
        T1 = t3-t2
        T2 = t4-t3
        T3 = t5-t4
        write_log(rank,spg, iteration_1, T0, T1, T2, T3, fail_2, fail_3, E, V)

if __name__ == '__main__':
    main()
