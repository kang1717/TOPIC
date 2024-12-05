from topic.topology_csp.module_structure \
        import generate_initial_structure_random, generate_initial_structure_shortest, get_space_group
from topic.topology_csp.module_lammps import lammps_write, pos_dict2cooall, \
        run_lj_lammps, coo2pos_dict, change_coo_index, run_lammps
from topic.topology_csp.module_scrutinize import check_topology
from topic.topology_csp.module_log \
        import write_log_head,make_poscars_contcars,write_log,write_poscars_contcars
from time import time
from mpi4py import MPI
import random
import os, sys, shutil, yaml
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.vasp import Poscar


def make_CONTCAR2structure(str_idx):
    FILE = 'CONTCAR_success_%s'%(str_idx)
    structure = Poscar.from_file(FILE).structure
    return structure

def make_textlist2structure(text):
    structure = Poscar.from_str(''.join(text)).structure
    return structure

def make_pos2structure(pos):
    lattice = Lattice(pos['latt'])
    structure = Structure(lattice, pos['atomarray'], pos['coor'], coords_are_cartesian=True)
    return structure

def check_candidates(total_yaml, candidates, structure_index, pos, E, sm, num_atom):
    if E > candidates['minimum_E'] + total_yaml['energy_window']*num_atom:
        return

    # Check duplicates
    structure1 = make_pos2structure(pos)
    new_key = 1
    for str_idx in candidates['structure'].keys():
        structure2 = candidates['structure'][str_idx]
        if sm.fit(structure1, structure2, symmetric=True):
            if E < candidates['E'][str_idx]:
                del candidates['structure'][str_idx]
                del candidates['E'][str_idx]
                candidates['structure'][structure_index] = structure1
                candidates['E'][structure_index] = E
                break
            else:
                return
            new_key = 0

    if new_key == 1:
        candidates['structure'][structure_index] = structure1
        candidates['E'][structure_index] = E

    # Update candidates if E is lower than previous lowest E
    del_key = []
    if E < candidates['minimum_E']:
        candidates['minimum_E'] = E
        for str_idx in candidates['E'].keys():
            if candidates['E'][str_idx] > candidates['minimum_E'] + total_yaml['energy_window']*num_atom:
                del_key.append(str_idx)

    for str_idx in del_key:
        del candidates['structure'][str_idx]
        del candidates['E'][str_idx]

    if len(candidates['E'].keys()) == 0:
        candidates['minimum_E'] = E
        candidates['structure'][structure_index] = structure1
        candidates['E'][structure_index] = E

    with open('BestStructure', 'w') as s:
        cand_list = sorted(candidates['E'].items(), key=lambda x:x[1])
        s.write("Index   Energy\n")
        for cand in cand_list:
            s.write("{:7} {:>8.3f}\n".format(cand[0], cand[1]))


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
        if total_yaml['continue'] == -1: # Automatically defined by checking last index
            if 'log' not in os.listdir('%s'%rank):
                start_idx = 0
            else:
                with open('%s/log'%rank, 'r') as f:
                    lines = f.readlines()
                    fin_idx = int(lines[-1].split()[1])
                start_idx = fin_idx + 1
        else: # Manually define # Manually define # Manually define
            start_idx = int(total_yaml['continue']/corenum)
    else:
        start_idx = 0
        total_yaml['continue'] = 0
    end_idx = int(total_yaml['generation']/corenum)


    num_atom = 0
    for elem in total_yaml['material'].keys():
        if elem != 'Li':
            num_atom += total_yaml['material'][elem]
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

    # Make candidate dictionary
    candidates = dict()
    candidates['minimum_E'] = 99999
    candidates['structure'] = dict()
    candidates['E'] = dict()
    if 'BestStructure' in os.listdir():
        with open('BestStructure', 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                if i == 1:
                    candidates['minimum_E'] = float(line.split()[1])
                index = int(line.split()[0])
                candidates['E'][index] = float(line.split()[1])

        indices = sorted(candidates['E'].keys())
        with open('CONTCAR3s', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'generation:' in line:
                    index = int(line.split()[4])
                    if index in indices:
                        text_list = [lines[i]]
                        idx = 1
                        while 'generation:' not in lines[i+idx]:
                            text_list.append(lines[i+idx])
                            idx += 1
                            if i+idx == len(lines):
                                break
                        candidates['structure'][index] = make_textlist2structure(text_list)
        for index in indices:
            if index not in candidates['structure'].keys():
                del candidates['E'][index]

    if 'structure_matcher_tolerance' in total_yaml.keys():
        tol = total_yaml['structure_matcher_tolerance']
        sm = StructureMatcher(ltol=tol, stol=tol, angle_tol=5, primitive_cell=False)
    else:
        sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)

    for i in range(start_idx, end_idx):
        E0 = E1 = E2 = 1000
        V0 = V1 = V2 = 0
        fail_0 = fail_1 = fail_2 = fail_3 = -1
        spg0 = spg1 = spg2 = spg3 = 0

        # 1. Generate initial structure
        t1 = time()
        if i%2 == 0: # Link oxygen in random order # Link oxygen in random order # Link oxygen in random order # Link oxygen in random order
            pos, bond_dict, trial, spg, spg0 = generate_initial_structure_random(total_yaml)
        else: # Link oxygen in shortest distance order
            pos, bond_dict, trial, spg, spg0 = generate_initial_structure_shortest(total_yaml)
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
                    check_candidates(total_yaml, candidates, i, pos, E2, sm, num_atom)
                    #with open(f'CONTCAR_success_{i}', 'w') as s:
                    #    s.write(contcar3_text)

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
