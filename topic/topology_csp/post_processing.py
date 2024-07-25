import os, shutil
import yaml
import chemparse
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser, CifWriter

import numpy as np
import sys, re
from pyzeo.netstorage import AtomNetwork
from pyzeo.extension import prune_voronoi_network_close_node
from lammps import lammps, PyLammps
from time import time

from mpi4py import MPI


atomic_mass = dict(H=1.01, He=4.00, Li=6.94, Be=9.01, B=10.81, C=12.01,
                   N=14.01, O=16.00, F=19.00, Ne=20.18, Na=22.99, Mg=24.31,
                   Al=26.98, Si=28.09, P=30.97, S=32.07, Cl=35.45, Ar=39.95,
                   K=39.10, Ca=40.08, Sc=44.96, Ti=47.87, V=50.94, Cr=52.00,
                   Mn=54.94, Fe=55.85, Co=58.93, Ni=58.69, Cu=63.55, Zn=65.39,
                   Ga=69.72, Ge=72.61, As=74.92, Se=78.96, Br=79.90, Kr=83.80,
                   Rb=85.47, Sr=87.62, Y=88.91, Zr=91.22, Nb=92.91, Mo=95.94,
                   Tc=98.00, Ru=101.07, Rh=102.91, Pd=106.42, Ag=107.87,
                   Cd=112.41, In=114.82, Sn=118.71, Sb=121.76, Te=127.60,
                   I=126.90, Xe=131.29, Cs=132.91, Ba=137.33, La=138.91,
                   Ce=140.12, Pr=140.91, Nd=144.24, Pm=145.00, Sm=150.36,
                   Eu=151.96, Gd=157.25, Tb=158.93, Dy=162.50, Ho=164.93,
                   Er=167.26, Tm=168.93, Yb=173.04, Lu=174.97, Hf=178.49,
                   Ta=180.95, W=183.84, Re=186.21, Os=190.23, Ir=192.22,
                   Pt=195.08, Au=196.97, Hg=200.59, Tl=204.38, Pb=207.2,
                   Bi=208.98, Po=209.00, At=210.00, Rn=222.00, Fr=223.00,
                   Ra=226.00, Ac=227.00, Th=232.04, Pa=231.04, U=238.03,
                   Np=237.00, Pu=244.00, Am=243.00, Cm=247.00, Bk=247.00,
                   Cf=251.00, Es=252.00, Fm=257.00, Md=258.00, No=259.00,
                   Lr=262.00, Rf=261.00, Db=262.00, Sg=266.00, Bh=264.00,
                   Hs=269.00, Mt=268.00)


def gather_candidates(n, inp):
    num_atom = 0
    for elem in inp['material'].keys():
        if elem != 'Li':
            num_atom += inp['material'][elem]

    if 'CONTCAR3s' in os.listdir(str(n)):
        with open(str(n)+'/CONTCAR3s','r') as f:
            lines = f.readlines()
            tot_num = int(len(lines)/(num_atom+8))

            idx = 1
            for i in range(tot_num):
                with open('poscars/POSCAR_%s_%s'%(n, idx), 'w') as s:
                    for k in range(8+num_atom):
                        s.write(lines[i*(8+num_atom)+k])
                idx += 1

def structure_match(FILE1, FILE2):
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)
    structure1 = Poscar.from_file(FILE1).structure
    structure1.remove_species(["Li"])
    structure2 = Poscar.from_file(FILE2).structure
    structure2.remove_species(["Li"])
    return sm.fit(structure1, structure2, symmetric=True)

def gather_unique_structure(DIR):
    poscars = os.listdir('poscars')

    unique_structure = []
    dup_dict = {n: [] for n in poscars}

    for i in range(len(poscars)-1):
        for j in range(i+1, len(poscars)):
            file1 = poscars[i]
            file2 = poscars[j]
            if structure_match(DIR+'/'+file1, DIR+'/'+file2): # If both are same
                dup_dict[file1].append(file2)
                dup_dict[file2].append(file1)
                if file1 not in unique_structure and file2 not in unique_structure:
                    unique_structure.append(file1)
            else: # If both are unique
                if file1 in unique_structure:
                    key1 = 0
                else:
                    key1 = all(file not in unique_structure for file in dup_dict[file1])
                if key1 == 1:
                    unique_structure.append(file1)

                if file2 in unique_structure:
                    key2 = 0
                else:
                    key2 = all(file not in unique_structure for file in dup_dict[file2])
                if key2 == 1:
                    unique_structure.append(file2)

    return unique_structure

# remove Li before calculate
def convert_poscar_to_cif(poscar_file):
    structure = Poscar.from_file(poscar_file).structure
    li_indices = [i for i, site in enumerate(structure) if site.species_string == 'Li']
    structure.remove_sites(li_indices)

    cif_file = 'tmp.cif'
    CifWriter(structure).write_file(cif_file)

    return cif_file

def convert_cif_to_cssr(cif_file):
    p = re.compile('[A-Z][a-z]?')
    p2 = re.compile(r'\d+')

    cssr_file = cif_file.replace('cif', 'cssr')
    cif = open(cif_file, 'r')
    cssr = open(cssr_file, 'w')

    lines = cif.readlines()
    lattice = [None]*6
    for i, line in enumerate(lines):
        if '_cell_length_a' in line:
            lattice[0] = line.split()[1]
        elif '_cell_length_b' in line:
            lattice[1] = line.split()[1]
        elif '_cell_length_c' in line:
            lattice[2] = line.split()[1]
        elif '_cell_angle_alpha' in line:
            lattice[3] = line.split()[1]
        elif '_cell_angle_beta' in line:
            lattice[4] = line.split()[1]
        elif '_cell_angle_gamma' in line:
            lattice[5] = line.split()[1]


        elif '_chemical_formula_structural' in line:
            name = line.split()[1]
            parse = chemparse.parse_formula(name)
            #elems = p.findall(name)
            #comps = p2.findall(name)
            elems = list(parse.keys())
            comps = list(parse.values())
            tot_atoms = sum(list(map(int, comps)))

            insert_line = '0 '
            for j in range(len(elems)):
                insert_line += '%s%s '%(elems[j], comps[j])
            insert_line += '\n'

        elif '_atom_site_occupancy' in line:
            # write lattice informations
            length = ' '.join(lattice[:3])
            angle = ' '.join(lattice[3:])
            cssr.write(length + '\n')
            cssr.write(angle + ' SPGR = 1 P 1   OPT = 1\n')

            # write composition informations
            cssr.write(str(tot_atoms) + ' 0\n')
            cssr.write(insert_line)

            #for idx in range(1, len(lines)):
            idx = 1
            while i+idx < len(lines):
                data = lines[i+idx].split()
                del data[1:3]
                del data[-1]
                cssr.write(str(idx+1)+' '+' '.join(data)+' 0 0 0 0 0 0 0 0 0.00\n')
                idx += 1
            break

    cssr.close()

    return cssr_file

def calculate_li_sites(poscar_file, li_file, max_li):
    # Convert POSCAR to CSSR file
    cif_file = convert_poscar_to_cif(poscar_file)
    cssr_file = convert_cif_to_cssr(cif_file)

    # Perform voronoi decomposition and calculate candidate Li sites
    atmnet = AtomNetwork.read_from_CSSR(cssr_file)
    vornet = prune_voronoi_network_close_node(atmnet, delta=2) # delta mean remove Li atoms that close to other Li atom
    vornet.write_to_XYZ(li_file)

    return cif_file

def concatenate_initial_li(cif_file, li_file, final_file, max_li):
    structure = CifParser(cif_file).parse_structures(primitive=False)[0]
    li_pos = []

    with open(li_file, 'r') as f:
        lines = f.readlines()
        natoms = int(lines.pop(0))
        lines.pop(0)
        for _ in range(natoms):
            line = lines.pop(0)
            x, y, z, rad = map(float, line.split()[1:5])
            li_pos.append([x, y, z, rad])

    if len(li_pos) < max_li:
        print("Number of Li candidate sites are less than {}".format(max_li))
        exit()

    li_pos = sorted(li_pos, key=lambda pos: pos[3], reverse=True)[:max_li]
    for pos in li_pos:
        structure.append("Li", pos[:3], coords_are_cartesian=True)

    Poscar(structure).write_file(final_file)

def relax(poscar, relaxed_file, lmp):
    potential = '../input/potentialli'

    poscar_data = Poscar.from_file(poscar)
    structure = poscar_data.structure
    elements = [el.symbol for el in structure.composition.keys()]
    relax_iter = 1

    os.system("/data/hswoo369/module/pos2lammps.sh {} > coo".format(poscar))

    lmp.command("clear")
    lmp.command("units           metal")
    lmp.command("newton          on")
    lmp.command("dimension       3")
    lmp.command("boundary        p p p")
    lmp.command("atom_style 	atomic")
    lmp.command("box  tilt large")
    lmp.command("read_data coo")
    lmp.command("pair_style nn/intel")
    lmp.command("pair_coeff * * %s %s"%(potential, ' '.join(elements)))
    for i, element in enumerate(elements, 1):
        lmp.command(f"mass {i} {atomic_mass[element]}")
    lmp.command("neigh_modify 	every 1 delay 0 check yes")
    lmp.command("neighbor 0.2 bin")
    lmp.command("compute 	_rg all gyration")

    lmp.command("run 0")
    lmp.command("variable e equal etotal")
    e1 = lmp.extract_variable("e")
    #e1 = lmp.variables['e'].value

    lmp.command('min_style cg')
    lmp.command("min_modify line quadratic dmax 0.1")
    lmp.command("minimize 0 0.02 %d 10000"%(int(relax_iter*len(structure))))
    lmp.command("minimize 0 0.02 %d 10000"%(int(relax_iter*len(structure))))
    lmp.command("minimize 0 0.02 %d 10000"%(int(relax_iter*len(structure))))

    lmp.command("min_modify      line quadratic dmax 0.1 ")
    lmp.command("fix 2 all box/relax tri 0.0 vmax 0.001")
    lmp.command("minimize 1.0e-6 0 %d 10000"%(relax_iter*len(structure)))
    lmp.command("unfix 2")

    lmp.command("fix 3 all box/relax tri 0.0 vmax 0.0001")
    lmp.command("min_modify      line quadratic dmax 0.1 ")
    lmp.command("minimize 1.0e-6 0 %d 10000"%(relax_iter*len(structure)))
    lmp.command("unfix 3")

    lmp.command('write_data coo_relaxed')

    lmp.command("run 0")
    lmp.command("variable e equal etotal")
    e1 = lmp.extract_variable("e")
    #e1 = lmp.variables['e'].value
    os.system("python3 /data/hswoo369/module/lammps2vasp_Direct.py coo_relaxed > {}".format(relaxed_file))

    return e1

def swap_li_site(poscar, li_file, new_poscar):
    structure = Poscar.from_file(poscar).structure
    li_indices = [i for i, site in enumerate(structure) if site.species_string == 'Li']
    all_positions = structure.cart_coords
    li_positions = all_positions[li_indices]

    candidate_positions = []
    with open(li_file, 'r') as f:
        lines = f.readlines()
        natoms = int(lines.pop(0))
        lines.pop(0)
        for _ in range(natoms):
            line = lines.pop(0)
            x, y, z, rad = map(float, line.split()[1:5])
            candidate_positions.append([x, y, z])

    # select one li position (while)
    for i in range(100):
        rnd_idx = np.random.choice(len(li_positions))
        selected_pos = li_positions[rnd_idx]

        # --- match selected Li site
        distances = np.linalg.norm(candidate_positions - selected_pos, axis=1)

        min_index = np.argmin(distances)
        matched_site = candidate_positions[min_index]
        candidate_positions = np.delete(candidate_positions, min_index, axis=0)
        distances = np.delete(distances, min_index, axis=0)

        # Define probability distribution
        beta = 1.0
        probabilities = np.exp(-beta * distances)
        probabilities /= probabilities.sum()

        # get neighbor of selected Li site
        random_index = np.random.choice(len(probabilities))
        new_site = candidate_positions[random_index]

        if np.min(np.linalg.norm(li_positions - new_site, axis=1)) > 2:
            idx = None
            for i, pos in enumerate(all_positions):
                if np.array_equal(pos, selected_pos):
                    idx = i
                    break

            all_positions[idx] = new_site
            structure.replace(idx, "Li", new_site, coords_are_cartesian=True)
            Poscar(structure).write_file(new_poscar)

            return selected_pos, new_site, i+1

    print("In trials, we cannot find swap positions")
    exit()

def log(file, message):
    with open(file, 'a') as s:
        s.write(message+'\n')

def mc_main(DIR, SAVE_DIR, file_name, max_li, max_mc_steps):
    shutil.copy(DIR+'/'+file_name, file_name)

    poscar_file = file_name
    # 1. Make Li inserted structure
    cif_file = calculate_li_sites(poscar_file, li_file='Li.xyz', max_li=max_li)
    concatenate_initial_li(cif_file, li_file='Li.xyz', final_file='POSCAR_init', max_li=max_li)

    # 2. NNP relax

    lmp = lammps('simd_serial', cmdargs=['-log', 'none', '-screen', 'none'])
    #lmp = lammps('simd_serial')

    prev_e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax', lmp=lmp)

    best_e = 999
    for step in range(max_mc_steps):
        # 3. Calculate candidate Li sites
        t2 = time()
        cif_file = calculate_li_sites(poscar_file='POSCAR_relax', li_file='Li.xyz', max_li=max_li)

        # 4. Randomly select one Li ions
        t3 = time()
        selected_pos, new_site, trials = swap_li_site(poscar='POSCAR_relax', li_file='Li.xyz', new_poscar='POSCAR_init')

        # 5. NNP relax
        t4 = time()
        e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax_tmp', lmp=lmp)

        t5 = time()
        log('log','Step: {:3>} E: {:7.3f} Swap_trials: {:3>} t_li_sites: {:8.5f} t_swap: {:8.5f} t_relax: {:8.5f}'.format(step, e, trials, t3-t2, t4-t3, t5-t4))

        # Postprocess
        if e < prev_e:
            accept = True
        else:
            accept = False

        if accept:
            os.rename('POSCAR_relax_tmp', 'POSCAR_relax')
            prev_e = e
            shutil.copy('POSCAR_relax', 'POSCAR_best')

    shutil.copy('POSCAR_best', SAVE_DIR+'/'+file_name)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open(sys.argv[1], 'r') as f:
        inp = yaml.safe_load(f)
        num_li = inp['material']['Li']
        max_mc_steps = inp['post_process']['max_mc_steps']

    # Filter duplicated structures
    if rank == 0:
        cwd = os.getcwd()
        if 'poscars' not in os.listdir('.'):
            os.mkdir('poscars')
        if 'unique_poscars' not in os.listdir('.'):
            os.mkdir('unique_poscars')
        if 'after_mc_poscars' not in os.listdir('.'):
            os.mkdir('after_mc_poscars')
        if 'final_poscars' not in os.listdir('.'):
            os.mkdir('final_poscars')

        for n in range(1000):
            if str(n) not in os.listdir():
                continue
            gather_candidates(n, inp)

        unique_structure = gather_unique_structure(DIR='poscars')

        for n in unique_structure:
            shutil.copy('poscars/'+n, 'unique_poscars/'+n)
    else:
        unique_structure = None

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
        if 'mc_%s'%rank not in os.listdir('.'):
            os.mkdir('mc_%s'%rank)
        os.chdir('mc_%s'%rank)
        for n in unique_structure[begin:end]:
            mc_main(DIR=cwd+'/unique_poscars', SAVE_DIR=cwd+'/after_mc_poscars', file_name=n, max_li=num_li, max_mc_steps=max_mc_steps)
        os.chdir('..')

    comm.barrier()

    if rank == 0:
        unique_structure = gather_unique_structure(DIR='after_mc_poscars')
        for n in unique_structure:
            shutil.copy('after_mc_poscars/'+n, 'final_poscars/'+n)

if __name__ == '__main__':
    main()

