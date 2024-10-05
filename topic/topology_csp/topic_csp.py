from topic.topology_csp.module_structure \
        import generate_initial_structure, get_space_group
from topic.topology_csp.module_lammps import lammps_write, pos_dict2cooall, \
        run_lj_lammps, coo2pos_dict, change_coo_index, run_lammps
from topic.topology_csp.module_scrutinize import check_topology
from topic.topology_csp.module_log \
        import write_log_head,make_poscars_contcars,write_log,write_poscars_contcars
from time import time
from mpi4py import MPI
import random
import os, sys, shutil, yaml


def main():
    input_file = str(sys.argv[1])
    with open(input_file, 'r') as f:
        total_yaml = yaml.safe_load(f)

    # MPI setting
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    corenum = comm.Get_size()

    ########### input #############
    if 'spg_seed' in total_yaml.keys():
        if total_yaml['spg_seed'] < 1 or total_yaml['spg_seed'] > 230:
            print('Wrong space group seed')
            exit()
    else:
        total_yaml['spg_seed'] = None

    if 'continue' in total_yaml.keys():
        start_idx = int(total_yaml['continue']/corenum)
    else:
        start_idx = 0
        total_yaml['continue'] = 0
    end_idx = int(total_yaml['generation']/corenum)
    ###############################

    if os.path.isdir(str(rank)) == False:
        shutil.copytree('input', str(rank))
    os.chdir(str(rank))

    if total_yaml['continue'] == 0:
        write_log_head()

    poscars = []
    contcars = []
    contcar2s = []
    contcar3s = []
    buffer = 0

    for i in range(start_idx, end_idx):
        E0 = E1 = E2 = 1000
        V0 = V1 = V2 = 0
        fail_0 = fail_1 = fail_2 = fail_3 = -1
        spg0 = spg1 = spg2 = spg3 = 0

        # 1. Generate initial structure
        t1 = time()
        pos, bond_dict, trial, spg, spg0 = generate_initial_structure(total_yaml)
        fail_0 = check_topology(total_yaml, pos)
        poscar_text = make_poscars_contcars(pos, rank, i)
        t2 = time()

        # 2. LJ & harmonic potential relax 
        lammps_write(pos, total_yaml, bond_dict)     # write lammps input: in.all
        pos_dict2cooall(pos, output='coo')
        E0, V0 = run_lj_lammps('in.all')
        if E0 == 0 and V0 == 0:
            continue

        pos = coo2pos_dict('coo_out', total_yaml)
        spg1 = get_space_group(pos)
        fail_1 = check_topology(total_yaml, pos)
        #if E0 != 0 or V0 != 0:
        #    pos = coo2pos_dict('coo_out', total_yaml)
        #    spg1 = get_space_group(pos)
        #    fail_1 = check_topology(total_yaml, pos)
        #else:
        #    continue

        buffer += 1
        poscars.append(poscar_text)
        contcar_text = make_poscars_contcars(pos, rank, i)
        contcars.append(contcar_text)
        t3 = time()

        # 3. Short NNP relax
        t4 = t3
        t5 = t3
        if fail_1 == 0:
            change_coo_index('coo_out', pos)
            E1, V1 = run_lammps('in.nnp')

            pos = coo2pos_dict('coo_nnp', total_yaml)
            spg2 = get_space_group(pos)
            fail_2 = check_topology(total_yaml, pos)

            contcar2_text = make_poscars_contcars(pos, rank, i)
            contcar2s.append(contcar2_text)
            t4 = time()

            # 4. Long NNP  relax
            t5 = t4
            if fail_2 == 0:
                E2, V2 = run_lammps('in.nnplong')

                pos = coo2pos_dict('coo_nnp2', total_yaml)
                spg3 = get_space_group(pos)
                fail_3 = check_topology(total_yaml, pos)

                contcar3_text = make_poscars_contcars(pos, rank, i)
                contcar3s.append(contcar3_text)
                t5 = time()

                if fail_3 == 0:
                    with open(f'CONTCAR_success_{i}', 'w') as s:
                        s.write(contcar3_text)

        # 5. Finalize
        T0 = t2-t1
        T1 = t3-t2
        T2 = t4-t3
        T3 = t5-t4
        write_log(rank, i, trial, fail_0, T0, spg, spg0, fail_1, T1, spg1, E0, V0,\
                                fail_2, T2, spg2, E1, V1, fail_3, T3, spg3, E2, V2)

        if buffer >= 10:
            write_poscars_contcars(poscars, contcars, contcar2s, contcar3s)
            poscars = []
            contcars = []
            contcar2s = []
            contcar3s = []
            buffer = 0

    write_poscars_contcars(poscars, contcars, contcar2s, contcar3s)

if __name__ == '__main__':
    main()
