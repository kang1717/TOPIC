import os, shutil
import yaml
import sys, re
from mpi4py import MPI

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher

from topic.topology_csp.module_mc import mc_main


def get_unique_file_list(total_yaml, comm, num_atom):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if len(os.listdir('unique_poscars')) != 0:
        if rank == 0:
            unique_structure = []
            for n in os.listdir('unique_poscars'):
                if n not in os.listdir('after_mc_poscars') and n not in os.listdir('ERROR'):
                    unique_structure.append(n)
            return unique_structure
        else:
            return None

    if rank == 0:
        # Concatenate all the BestStructure file and re-filtering within E window
        energy_info = dict()
        for i in range(0, 100):
            if str(i) not in os.listdir():
                continue
            if 'BestStructure' not in os.listdir(str(i)):
                continue
            with open('%s/BestStructure'%(i), 'r') as f:
                for ii, line in enumerate(f.readlines()):
                    if ii == 0:
                        continue
                    tmp = line.split()
                    index = int(tmp[0])
                    energy = float(tmp[1])
                    energy_info['POSCAR_%s_%s'%(i, index)] = energy

        min_e = min(energy_info.values())
        del_key = []
        for key in energy_info.keys():
            if energy_info[key] > min_e + total_yaml['energy_window']*num_atom:
                del_key.append(key)
        for key in del_key:
            del energy_info[key]

        # Sort in energy order and collect structure object
        candidate_dict = dict()
        for name in energy_info.keys():
            tmp = name.split('_')
            dnum = int(tmp[1])
            gen = int(tmp[2])
            if dnum not in candidate_dict.keys():
                candidate_dict[dnum] = []
            candidate_dict[dnum].append(gen)

        poscar_dict = dict()
        for i in range(0, 100):
            if str(i) not in os.listdir():
                continue
            if i not in candidate_dict.keys():
                continue
            with open('%s/CONTCAR3s'%i, 'r') as f:
                lines = f.readlines()
                for ii, line in enumerate(lines):
                    if 'generation' in line:
                        gen = int(line.split()[4])
                        if gen in candidate_dict[i]:
                            poscar = lines[ii:ii+num_atom+8]
                            poscar_dict['POSCAR_%s_%s'%(i, gen)] = poscar

        structure_list = sorted(energy_info.items(), key=lambda x:x[1])
        structures = []
        for n in structure_list:
            structure = Structure.from_str("".join(poscar_dict[n[0]]), fmt='poscar')
            structure.remove_species(["Li"])
            structures.append((n[0], n[1], structure))
    else:
        structures = None

    structures = comm.bcast(structures, root=0)
    if 'structure_matcher_tolerance' in total_yaml.keys():
        tol = total_yaml['structure_matcher_tolerance']
        sm = StructureMatcher(ltol=tol, stol=tol, angle_tol=5, primitive_cell=False)
    else:
        sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)

    prev_div = 99999999999999
    current_div = len(structures)/size
    trial = 1
    while prev_div/current_div >= 1.5:
        totn = len(structures)
        q = totn // size
        r = totn % size
        begin = rank * q + min(rank, r)
        end = begin + q
        if r > rank:
            end += 1

        loc_str_list = []
        if end - begin > 200:
            idx = begin + 200
            while idx < end:
                loc_str_list.append(structures[idx-200:idx])
                idx += 200
            if end > idx-200:
                loc_str_list.append(structures[idx-200:end])
        else:
            loc_str_list.append(structures[begin:end])

        unique_structures = []
        for local_structure in loc_str_list:
            unique_structures += gather_unique_structure(local_structure, rank, sm)

        all_unique_structures = comm.gather(unique_structures, root=0)

        if rank == 0:
            structures = []
            for local_res in all_unique_structures:
                structures += local_res
        else:
            structures = None

        structures = comm.bcast(structures, root=0)
        prev_div = current_div
        current_div = len(structures)/size
        trial += 1

    if rank == 0:
        if len(structures) <= 100:
            structures = gather_unique_structure(structures, rank, sm)

        for unique in structures:
            n = unique[0]
            tmp = n.split('_')
            with open('unique_poscars/%s'%(n), 'w') as s:
                s.write("".join(poscar_dict[n]))

        unique_structure = []
        for n in os.listdir('unique_poscars'):
            if n not in os.listdir('after_mc_poscars'):
                unique_structure.append(n)

        return unique_structure
    else:
        return None

def gather_unique_structure(structures, rank, sm):
    unique_structures = []
    unique_structure_name = []

    for filename, energy, structure in structures:
        if not any(sm.fit(structure, uniq_struct) for _, energy2, uniq_struct in unique_structures):
            unique_structures.append((filename, energy, structure))

    return unique_structures

def gather_unique_structure_from_DIR(total_yaml, DIR):
    # Collect with in energy window structures
    energy_info = dict()
    for i in range(0, 100):
        if 'mc_'+str(i) not in os.listdir():
            continue
        if 'Results' not in os.listdir('mc_'+str(i)):
            continue
        with open('mc_%s/Results'%(i), 'r') as f:
            for ii, line in enumerate(f.readlines()):
                tmp = line.split()
                file_name = tmp[0]
                energy = float(tmp[1])
                energy_info[file_name] = energy

    min_e = min(energy_info.values())
    del_key = []
    for key in energy_info.keys():
        if energy_info[key] > min_e + total_yaml['mc_energy_window']:
            del_key.append(key)
    for key in del_key:
        del energy_info[key]

    # Check the duplicates (only with host structure)
    structures = []
    for n in energy_info.keys():
        structure = Poscar.from_file(DIR+'/'+n).structure
        structure.remove_species(["Li"])
        structures.append((n, structure))

    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)
    unique_structures = []
    unique_filenames = []

    for filename, structure in structures:
        if not any(sm.fit(structure, uniq_struct) for _, uniq_struct in unique_structures):
            unique_structures.append((filename, structure))
            unique_filenames.append(filename)

    return unique_filenames

def log(file, message):
    with open(file, 'a') as s:
        s.write(message+'\n')

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open(sys.argv[1], 'r') as f:
        inp = yaml.safe_load(f)
        num_li = inp['material']['Li']
        natoms = sum(list(map(int, inp['material'].values())))
        num_host = natoms - num_li

        post_process_type = inp['post_process']['type']
        if post_process_type == 'mc':
            max_mc_steps = inp['post_process']['max_mc_steps']
            T = inp['post_process']['mc_temperature']
        elif post_process_type == 'md':
            max_md_steps = inp['post_process']['max_md_steps']
            T = inp['post_process']['md_temperature']

    # Filter duplicated structures
    if rank == 0:
        cwd = os.getcwd()
        os.makedirs('unique_poscars', exist_ok=True)
        os.makedirs('after_mc_poscars', exist_ok=True)
        os.makedirs('final_poscars', exist_ok=True)
        os.makedirs('ERROR', exist_ok=True)
    comm.Barrier()

    unique_structure = get_unique_file_list(inp, comm, num_host)
    unique_structure = comm.bcast(unique_structure, root=0)

    # Run MC for all candidates
    tot_len = len(unique_structure)

    q = tot_len // size
    r = tot_len % size

    if rank < r:
        begin = rank * (q+1)
        end = begin + q + 1
    else:
        begin = rank * q + r
        end = begin + q
    cwd = os.getcwd()

    if rank < tot_len:
        if post_process_type == 'mc':
            if 'mc_%s'%rank not in os.listdir('.'):
                os.mkdir('mc_%s'%rank)
            os.chdir('mc_%s'%rank)
            log('log','{:16}  {:4}  {:8}  {:6}  {:4}  {:10}  {:5}  {:6}  {:7}  {:3}'.format("File name", "Step", "Energy", "Accept", "Fail", "Swap trial", "t_zeo", "t_swap", "t_relax", "spg"))
            for n in unique_structure[begin:end]:
                mc_main(inp, DIR=cwd+'/unique_poscars', SAVE_DIR=cwd+'/after_mc_poscars', file_name=n, max_li=num_li, natoms=natoms, max_mc_steps=max_mc_steps, T=T)

        os.chdir('..')

    comm.barrier()

    if rank == 0:
        unique_structure = gather_unique_structure_from_DIR(inp, DIR='after_mc_poscars')
        for n in unique_structure:
            shutil.copy('after_mc_poscars/'+n, 'final_poscars/'+n)

if __name__ == '__main__':
    main()

