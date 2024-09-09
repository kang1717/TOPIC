import os, shutil
import yaml
import chemparse
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifParser, CifWriter
from topic.topology_csp.basic_tools import read_poscar_dict, calculate_distance

import numpy as np
import sys, re
from pyzeo.netstorage import AtomNetwork
from pyzeo.extension import prune_voronoi_network_close_node
from lammps import lammps, PyLammps
from time import time

from mpi4py import MPI


anion_type = ['O'] #,'F','S','Cl']

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

    for m in os.listdir(str(n)):
        if 'CONTCAR_success' in m:
            idx = m.split('_')[2]
            shutil.copy(str(n)+'/'+m, 'poscars/POSCAR_%s_%s'%(n, idx))
    """
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
    """

def structure_match(FILE1, FILE2):
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False)
    structure1 = Poscar.from_file(FILE1).structure
    structure1.remove_species(["Li"])
    structure2 = Poscar.from_file(FILE2).structure
    structure2.remove_species(["Li"])
    return sm.fit(structure1, structure2, symmetric=True)

def gather_unique_structure(DIR):
    structures = []
    structure_files = os.listdir(DIR)
    for n in structure_files:
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

def concatenate_initial_li(inp, cif_file, li_file, final_file, max_li):
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

    if inp['post_process']['initial_li_sites'] == 'random':
        li_pos = np.array(li_pos)
        rnd_idx = np.random.choice(len(li_pos), max_li, False)
        li_pos = li_pos[rnd_idx]
    elif inp['post_process']['initial_li_sites'] == 'free_spcae':
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
            for j, pos in enumerate(all_positions):
                if np.array_equal(pos, selected_pos):
                    idx =j 
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

def md_main(inp, DIR, SAVE_DIR, file_name, max_li, natoms, max_md_steps, T):
    shutil.copy(DIR+'/'+file_name, file_name)

    poscar_file = file_name
    # 1. Make Li inserted structure
    t1 = time()
    cif_file = calculate_li_sites(poscar_file, li_file='Li.xyz', max_li=max_li)
    concatenate_initial_li(inp, cif_file, li_file='Li.xyz', final_file='POSCAR_init', max_li=max_li)
    shutil.copy('POSCAR_init', poscar_file+'_init')

    # 2. NNP relax
    t2 = time()
    lmp = lammps('simd_serial', cmdargs=['-log', 'none', '-screen', 'none'])
    #lmp = lammps('simd_serial')

    #relax_e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax', lmp=lmp)
    #t3 = time()
    #shutil.copy('POSCAR_relax', poscar_file+'_relax')
    #tot_e = md(poscar_file+"_relax", poscar_file+"_fin", lmp, max_md_steps, T)
    tot_e = md(poscar_file+"_init", poscar_file+"_fin", lmp, max_md_steps, T)
    t3 = time()
    fail = check_topology(inp, poscar=poscar_file+"_fin")
    log('log','{:12}  {:4}  {:6.2f}  {:5.2f}  {:7.2f}'.format(file_name, fail, tot_e, t2-t1, t3-t2))

def md(poscar, relaxed_file, lmp, max_md_steps, T):
    potential = '../input/potentialli'

    f = open(poscar, 'r')
    lines = f.readlines()
    f.close()
    tot_num = sum(list(map(int, lines[6].split())))
    elems = lines[5].split()
    relax_iter = 1

    os.system("/data/hswoo369/module/pos2lammps.sh {} > coo".format(poscar))

    #lmp = lammps('simd_serial')
    #lmp = PyLammps('simd_serial', verbose=False)

    lmp.command("clear")
    lmp.command("units           metal")
    lmp.command("newton          on")
    lmp.command("dimension       3")
    lmp.command("boundary        p p p")
    lmp.command("atom_style 	atomic")
    lmp.command("box  tilt large")
    lmp.command("read_data coo")
    lmp.command("pair_style nn/intel")
    lmp.command("pair_coeff * * %s %s"%(potential, lines[5].strip()))
    lmp.command("mass 1 %s"%atomic_mass[elems[0]])
    lmp.command("mass 2 %s"%atomic_mass[elems[1]])
    lmp.command("mass 3 %s"%atomic_mass[elems[2]])
    lmp.command("mass 4 %s"%atomic_mass[elems[3]])
    lmp.command("neigh_modify 	every 1 delay 0 check yes")
    lmp.command("neighbor 0.2 bin")

    lmp.command("velocity all create %s 1 mom yes rot yes dist gaussian"%T)
    pressure = 0
    lmp.command("fix 1 all npt temp %s %s 0.1 x %s %s 1.0 y 0.0 0.0 1.0 z 0.0 0.0 1.0 couple none"%(T, T, pressure, pressure))

    lmp.command("timestep 0.001")
    lmp.command("dump traj all custom 10 out_%s.xyz id type x y z fx fy fz"%poscar)
    lmp.command("dump_modify traj sort id")
    lmp.command("thermo 100")
    lmp.command("thermo_style custom step temp press vol density etotal")
    lmp.command("run %s"%max_md_steps)

    lmp.command('write_data coo_relaxed')

    lmp.command("variable e equal etotal")
    e1 = lmp.extract_variable("e")
    #e1 = lmp.variables['e'].value
    os.system("python3 /data/hswoo369/module/lammps2vasp_Direct.py coo_relaxed > {}".format(relaxed_file))

    return e1

def check_topology(inp, poscar):
    cation_cn = inp['cation_cn']
    bond_dict = inp['distance_constraint']
    scrutinize_factor = inp['scrutinize_distance_factor']

    pos = read_poscar_dict(poscar)
    if pos['cartesian'] == False:
        pos['coor'] = np.dot(pos['coor'], pos['latt'])

    # choose cation dict, anion dict
    n_cation = []
    n_anion  = []
    cation_dict = {}
    anion_dict  = {}

    for i,a in enumerate(pos['atomarray']):
        if a not in anion_type+['Li']:
            n_cation.append(i)
            cation_dict[i] = []
        elif a in anion_type:
            n_anion.append(i)
            anion_dict[i] = []

    # write a cation_dict of its own
    for c in n_cation:
        atom_c = pos['atomarray'][c]

        for a in n_anion:
            distance = calculate_distance(pos['coor'][c],pos['coor'][a],pos['latt'])
            atom_a = pos['atomarray'][a]

            bond_cut = bond_dict[atom_c+"-"+atom_a]

            if distance < bond_cut * scrutinize_factor:
                cation_dict[c].append(a)

    for cation in cation_dict:
        cation_dict[cation] = sorted(cation_dict[cation])

    fail = 0
    for ckey in cation_dict.keys():
        if len(cation_dict[ckey]) != cation_cn[pos['atomarray'][ckey]]:
            fail += 1

    return fail

def mc_main(inp, DIR, SAVE_DIR, file_name, max_li, natoms, max_mc_steps, T):
    shutil.copy(DIR+'/'+file_name, file_name)

    poscar_file = file_name
    # 1. Make Li inserted structure
    t2 = time()
    cif_file = calculate_li_sites(poscar_file, li_file='Li.xyz', max_li=max_li)

    inp['post_process']['initial_li_sites'] = 'free_space'
    concatenate_initial_li(inp, cif_file, li_file='Li.xyz', final_file='POSCAR_init', max_li=max_li)
    shutil.copy('POSCAR_init', poscar_file+'_0_init')

    t3 = time()
    # 2. NNP relax
    lmp = lammps('simd_serial', cmdargs=['-log', 'none', '-screen', 'none'])
    #lmp = lammps('simd_serial')
    t4 = time()

    prev_e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax', lmp=lmp) / natoms
    fail = check_topology(inp, poscar='POSCAR_relax')
    log('log','{:12}  {:4}  {:6.2f}  {:6}  {:4}  {:10}  {:5.2f}  {:6}  {:7.2f}'.format(file_name, 'Free', prev_e*natoms, '-', fail, '-', t3-t2, '-', t4-t3))


    if fail != 0:
        inp['post_process']['initial_li_sites'] = 'random'
        for trial in range(10):
            concatenate_initial_li(inp, cif_file, li_file='Li.xyz', final_file='POSCAR_init', max_li=max_li)
            shutil.copy('POSCAR_init', poscar_file+'_0_init')

            # 2. NNP relax
            t3 = time()
            lmp = lammps('simd_serial', cmdargs=['-log', 'none', '-screen', 'none'])
            #lmp = lammps('simd_serial')
            t4 = time()

            prev_e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax', lmp=lmp) / natoms
            fail = check_topology(inp, poscar='POSCAR_relax')
            log('log','{:12}  {:4}  {:6.2f}  {:6}  {:4}  {:10}  {:5}  {:6}  {:7.2f}'.format(file_name, 'Rand', prev_e*natoms, '-', fail, trial+1, '-', '-', t4-t3))
            if fail == 0:
                break
        if fail != 0:
            log('log','{:12} fail to generate initial Li distribution'.format(file_name))
            return

    shutil.copy('POSCAR_relax', poscar_file+'_0_relax')


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
        tot_e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax_tmp', lmp=lmp)
        e = tot_e / natoms

        t5 = time()

        # Postprocess
        if e < prev_e:
            accept = True
        else:
            if T == 0:
                accept = False
            else:
                ratio = np.exp(-(e - prev_e)/0.025851 * 300/T)
                if np.random.rand() < ratio:
                    accept = True
                else:
                    accept = False

        fail = check_topology(inp, poscar='POSCAR_relax_tmp')
        if fail != 0:
            accept = False

        log('log','{:12}  {:4}  {:6.2f}  {:6}  {:4}  {:10}  {:5.2f}  {:6.2f}  {:7.2f}'.format(file_name, step, tot_e, accept, fail, trials, t3-t2, t4-t3, t5-t4))

        if accept:
            #os.rename('POSCAR_relax_tmp', 'POSCAR_relax')
            os.rename('POSCAR_relax_tmp', poscar_file+'_%s_relax'%(step+1))
            prev_e = e
            #shutil.copy('POSCAR_relax', 'POSCAR_best')
            shutil.copy(poscar_file+'_%s_relax'%(step+1), poscar_file+'_best')

    if poscar_file+'_best' in os.listdir():
        os.rename(poscar_file+'_best', SAVE_DIR+'/'+file_name)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with open(sys.argv[1], 'r') as f:
        inp = yaml.safe_load(f)
        num_li = inp['material']['Li']
        natoms = sum(list(map(int, inp['material'].values())))
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
        if 'poscars' not in os.listdir('.'):
            os.mkdir('poscars')
        if 'unique_poscars' not in os.listdir('.'):
            os.mkdir('unique_poscars')
        if 'after_mc_poscars' not in os.listdir('.'):
            os.mkdir('after_mc_poscars')
        if 'final_poscars' not in os.listdir('.'):
            os.mkdir('final_poscars')

        if len(os.listdir('unique_poscars')) != 0:
            unique_structure = []
            for n in os.listdir('unique_poscars'):
                if n not in os.listdir('after_mc_poscars'):
                    unique_structure.append(n)
        else:
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
        if post_process_type == 'mc':
            if 'mc_%s'%rank not in os.listdir('.'):
                os.mkdir('mc_%s'%rank)
            os.chdir('mc_%s'%rank)
            log('log','{:12}  {:4}  {:6}  {:6}  {:4}  {:10}  {:5}  {:6}  {:7}'.format("File name", "Step", "Energy", "Accept", "Fail", "Swap trial", "t_zeo", "t_swap", "t_relax"))
            for n in unique_structure[begin:end]:
                mc_main(inp, DIR=cwd+'/unique_poscars', SAVE_DIR=cwd+'/after_mc_poscars', file_name=n, max_li=num_li, natoms=natoms, max_mc_steps=max_mc_steps, T=T)
        elif post_process_type == 'md':
            if 'md_%s'%rank not in os.listdir('.'):
                os.mkdir('md_%s'%rank)
            os.chdir('md_%s'%rank)
            log('log','{:12}  {:4}  {:6}  {:5}  {:7}'.format("File name", "Fail", "TotalE", "t_zeo", "t_md"))
            for n in unique_structure[begin:end]:
                md_main(inp, DIR=cwd+'/unique_poscars', SAVE_DIR=cwd+'/after_mc_poscars', file_name=n, max_li=num_li, natoms=natoms, max_md_steps=max_md_steps, T=T)

        os.chdir('..')

    comm.barrier()

    if rank == 0:
        unique_structure = gather_unique_structure(DIR='after_mc_poscars')
        for n in unique_structure:
            shutil.copy('after_mc_poscars/'+n, 'final_poscars/'+n)

if __name__ == '__main__':
    main()

