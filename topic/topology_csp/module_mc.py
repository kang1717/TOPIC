import os, shutil, sys, re
from time import time

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN

from lammps import lammps
from topic.topology_csp.basic_tools import calculate_distance
from topic.topology_csp.module_scrutinize import check_topology
from topic.topology_csp.module_structure import get_space_group
from topic.topology_csp.module_lammps import coo2pos_dict_host


script_dir = os.path.dirname(os.path.abspath(__file__))

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

def lammps2DIRECT(lammps_file, poscar_file):
    LammpsLines = open(lammps_file).readlines()

    #------------------------- simple structure file part -----------------------#
    LatticeVector = [[0,0,0],[0,0,0],[0,0,0]]
    name = LammpsLines[0]
    for cnt in range(1,len(LammpsLines)):
        if 'atoms' in LammpsLines[cnt]:
            numIon = int(LammpsLines[cnt].split()[0])
        elif 'atom types' in LammpsLines[cnt]:
            numElements = int(LammpsLines[cnt].split()[0])
        elif 'xlo' in LammpsLines[cnt]:
            LatticeVector[0][0] = float(LammpsLines[cnt].split()[1]) -float(LammpsLines[cnt].split()[0])
        elif 'ylo' in LammpsLines[cnt]:
            LatticeVector[1][1] = float(LammpsLines[cnt].split()[1]) - float(LammpsLines[cnt].split()[0])
        elif 'zlo' in LammpsLines[cnt]:
            LatticeVector[2][2] = float(LammpsLines[cnt].split()[1]) -float(LammpsLines[cnt].split()[0])
        elif 'xy' in LammpsLines[cnt]:
            LatticeVector[1][0] = float(LammpsLines[cnt].split()[0])
            LatticeVector[2][0] = float(LammpsLines[cnt].split()[1])
            LatticeVector[2][1] = float(LammpsLines[cnt].split()[2])
        elif 'Masses' in LammpsLines[cnt]:
            MassesLines = []
            cursor = 0
            while True:
                cursor += 1
                if LammpsLines[cnt+cursor].split() != []:
                    MassesLines.append(LammpsLines[cnt+cursor])
                if len(MassesLines) == numElements:
                    break
        elif 'Atoms' in LammpsLines[cnt]:
            AtomsLines = []
            cursor = 0
            while True:
                cursor += 1
                if LammpsLines[cnt+cursor].split() != []:
                    AtomsLines.append(LammpsLines[cnt+cursor])
                if len(AtomsLines) == numIon:
                    break

    AtomsLines.sort(key=lambda x: float(x.split()[4]))   # sort about z coordinate
    AtomsLines.sort(key=lambda x: int(x.split()[1]))
    ElementList = []
    AMItems = list(atomic_mass.items())
    for line in MassesLines:
        mass = round(float(line.split()[1]),0)
        for Tup in AMItems:
            if round(Tup[1],0) == mass:
                ElementList.append(Tup[0])
                break

    NumIonList = [0]*numElements
    Direct=[]
    for line in AtomsLines:
        NumIonList[int(line.split()[1])-1] += 1
        cart = list(map(float,line.split()[2:]))
        frac = [0,0,0]
        frac[2] = cart[2]/LatticeVector[2][2]
        cart[1] -= frac[2]*LatticeVector[2][1]
        cart[0] -= frac[2]*LatticeVector[2][0]
        frac[1] = cart[1]/LatticeVector[1][1]
        cart[0] -= frac[1]*LatticeVector[1][0]
        frac[0] = cart[0]/LatticeVector[0][0]
        Direct.append(frac)

    printBuffer = ''
    printBuffer += '(Converted from LAMMPS) '+name+' 1.0\n'
    for rowCnt in range(3):
        for colCnt in range(3):
            printBuffer += '   %12.12f ' %float(LatticeVector[rowCnt][colCnt])
        printBuffer += '\n'
    for comp in ElementList:
        printBuffer += '  %s' %comp
    printBuffer += '\n'
    for comp in NumIonList:
        printBuffer += '  %s' %comp
    printBuffer += '\nDirect\n'
    for rowCnt in range(numIon):
        for colCnt in range(3):
            printBuffer += '   %12.12f ' %float(Direct[rowCnt][colCnt])
        printBuffer += '\n'
    with open(poscar_file, 'w') as s:
        s.write(printBuffer)

def cluster_atoms(positions, cell, cutoff=1.0):
    """
    Cluster atoms within the cutoff distance under PBC using DBSCAN.
    """
    # Perform clustering using DBSCAN
    db = DBSCAN(eps=cutoff, min_samples=1, metric='euclidean')  # Precomputed metric for distances
    labels = db.fit_predict(positions)
    return labels

def compute_cluster_averages(positions, labels):
    """
    Compute average position for each cluster.
    """
    clusters = {}
    for label in np.unique(labels):
        cluster_positions = positions[labels == label]
        average_position = np.mean(cluster_positions, axis=0)
        clusters[label] = average_position
    return clusters

def replicate_positions(positions, cell, reps):
    replicated_positions = []
    for i in range(-reps, reps+1):
        for j in range(-reps, reps+1):
            for k in range(-reps, reps+1):
                shift = i*cell[0] + j*cell[1] + k*cell[2]
                replicated_positions.extend(positions+shift)
    return np.array(replicated_positions)

def filter_vertices_within_cell(vertices, cell):
    inv_cell = np.linalg.inv(cell.T)
    fractional_position = np.dot(vertices, inv_cell)
    in_cell = np.all((fractional_position >= 0) & (fractional_position < 1), axis=1)
    return vertices[in_cell]

def calculate_li_sites(poscar_file, input_yaml):
    li_li_cutoff = input_yaml['post_process']['tolerance_Li_Li']
    li_node_cutoff = input_yaml['post_process']['tolerance_Li_node']

    structure = Structure.from_file(poscar_file)
    cell = structure.lattice.matrix
    o_positions = [site.coords for i, site in enumerate(structure) if (site.species_string == 'O')]

    reps = 1
    extended_o_positions = replicate_positions(o_positions, cell, reps)
    vor = Voronoi(extended_o_positions)
    voronoi_vertices = vor.vertices

    labels = cluster_atoms(voronoi_vertices, cell, cutoff=li_li_cutoff)
    cluster_averages = compute_cluster_averages(voronoi_vertices, labels)
    cluster_averages = np.array(list(cluster_averages.values()))
    filtered_vertices = filter_vertices_within_cell(cluster_averages, cell)

    metal_sites = [site.coords for i, site in enumerate(structure) if (site.species_string not in ['Li', 'O'])]
    li_pos = []
    for li_site in filtered_vertices:
        is_metal_site = False
        for site in metal_sites:
            d = calculate_distance(li_site, site, structure.lattice.matrix)
            if d <= li_node_cutoff:
                is_metal_site = True
                break

        distances = cdist([li_site], extended_o_positions)
        rad = np.min(distances)

        if is_metal_site == False:
            li_pos.append(np.append(li_site, rad))

    return structure, li_pos

def concatenate_initial_li(inp, structure, li_pos, final_file, max_li):
    material = inp['material']
    host_info = {k: v for k, v in material.items() if k not in {'Li', 'O'}}
    host_info = sorted(host_info.items(), key=lambda item: item[1], reverse=True)
    chem_order = [item[0] for item in host_info]
    chem_order.append('O')
    chem_order.append('Li')

    if len(li_pos) < max_li:
        return 1

    if inp['post_process']['initial_li_sites'] == 'random':
        li_pos = np.array(li_pos)
        rnd_idx = np.random.choice(len(li_pos), max_li, False)
        li_pos = li_pos[rnd_idx]
    elif inp['post_process']['initial_li_sites'] == 'free_space':
        li_pos = sorted(li_pos, key=lambda pos: pos[3], reverse=True)[:max_li]

    tmp_structure = structure.copy()
    for pos in li_pos:
        tmp_structure.append("Li", pos[:3], coords_are_cartesian=True)

    ordered_sites = []
    for elem in chem_order:
        ordered_sites.extend([site for site in tmp_structure if site.species_string == elem])
    custom_ordered_structure = Structure.from_sites(ordered_sites)
    Poscar(custom_ordered_structure).write_file(final_file)
    return 0

def relax(poscar, relaxed_file, lmp):
    potential = '../input/potentialli'

    poscar_data = Poscar.from_file(poscar)
    structure = poscar_data.structure
    elements = [el.symbol for el in structure.composition.keys()]
    relax_iter = 1

    os.system("{}/pos2lammps.sh {} > coo".format(script_dir, poscar))

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
    lmp.command("atom_modify sort 0 0.0")
    lmp.command("neigh_modify 	every 1 delay 0 check yes")
    lmp.command("neighbor 0.2 bin")
    lmp.command("compute 	_rg all gyration")

    lmp.command("run 0")
    lmp.command("variable e equal etotal")
    e1 = lmp.extract_variable("e")

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
    lammps2DIRECT('coo_relaxed', relaxed_file)

    return e1

def swap_li_site(structure, li_sites, new_poscar):
    tmp_structure = structure.copy()
    li_indices = [i for i, site in enumerate(tmp_structure) if site.species_string == 'Li']
    all_positions = tmp_structure.cart_coords
    li_positions = all_positions[li_indices]

    candidate_positions = []
    for site in li_sites:
        candidate_positions.append(site[:3])

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
        for ii in range(100):
            random_index = np.random.choice(len(probabilities))
            new_site = candidate_positions[random_index]

            if np.min(np.linalg.norm(li_positions - new_site, axis=1)) > 2:
                idx = None
                for j, pos in enumerate(all_positions):
                    if np.array_equal(pos, selected_pos):
                        idx = j
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

def extract_POSCAR_text_lines(poscar_name, step, status):
    with open(poscar_name, 'r') as f:
        text = f.readlines()
        text[0] = 'Step: %s %s\n'%(step, status)
    return text

def make_li_inserted_structure(inp, lmp, file_name, structure, li_sites, max_li, natoms, mode):
    fail = concatenate_initial_li(inp, structure, li_sites, final_file='POSCAR_init', max_li=max_li)
    if fail == 1:
        log('log','{:16} Number of Li candidate sites are less than {}'.format(file_name, max_li))
        return

    t3 = time()
    prev_e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax', lmp=lmp) / natoms
    t4 = time()

    pos = coo2pos_dict_host('coo_relaxed', inp)
    spg = get_space_group(pos)
    fail = check_topology(inp, pos)
    if fail == 0:
        accept = 1
    else:
        accept = 0
    log('log','{:16}  {:4}  {:8.4f}  {:6}  {:4}  {:10}  {:5}  {:6}  {:7.2f}  {:3}'.format(file_name, mode, prev_e, accept, fail, '-', '-', '-', t4-t3, spg))

    return fail, prev_e

def mc_main(inp, DIR, SAVE_DIR, file_name, max_li, natoms, max_mc_steps, T):
    poscar_file = DIR+'/'+file_name
    POSCAR_text = []

    # 1. Make Li inserted structure in the largest free space sites
    t2 = time()
    structure, li_sites = calculate_li_sites(poscar_file, inp)
    t3 = time()

    lmp = lammps('simd_serial', cmdargs=['-log', 'none', '-screen', 'none'])
    #lmp = lammps('simd_serial')

    inp['post_process']['initial_li_sites'] = 'free_space'
    fail, prev_e =  make_li_inserted_structure(inp, lmp, file_name, structure, li_sites, max_li, natoms, mode='Free')

    # 2. Make Li inserted structure in the random sites (10 trials)
    if fail != 0:
        inp['post_process']['initial_li_sites'] = 'random'
        for trial in range(10):
            fail, prev_e =  make_li_inserted_structure(inp, lmp, file_name, structure, li_sites, max_li, natoms, mode='Rand')
            if fail == 0:
                break
        if fail != 0:
            log('log','{:16} fail to generate initial Li distribution'.format(file_name))
            shutil.copy('../unique_poscars/'+file_name, '../ERROR/'+file_name)
            return

    shutil.copy('POSCAR_relax', 'POSCAR_best')
    POSCAR_text.append(extract_POSCAR_text_lines('POSCAR_init', 0, 'init'))
    POSCAR_text.append(extract_POSCAR_text_lines('POSCAR_relax', 0, 'relax'))

    best_e = prev_e
    # 3. Progressing MC algorithm 
    for step in range(max_mc_steps):
        t2 = time()
        structure, li_sites = calculate_li_sites('POSCAR_relax', inp)
        t3 = time()
        selected_pos, new_site, trials = swap_li_site(structure, li_sites, new_poscar='POSCAR_init')
        t4 = time()
        tot_e = relax(poscar='POSCAR_init', relaxed_file='POSCAR_relax_tmp', lmp=lmp)
        e = tot_e / natoms
        t5 = time()

        # Postprocess
        accept = False
        if e < prev_e:
            accept = True
        elif T > 0:
            ratio = np.exp(-(e - prev_e)/0.025851 * 300/T)
            if np.random.rand() < ratio:
                accept = True

        pos = coo2pos_dict_host('coo_relaxed', inp)
        spg = get_space_group(pos)
        fail = check_topology(inp, pos)
        if fail != 0:
            accept = False
        log('log','{:16}  {:4}  {:8.4f}  {:6}  {:4}  {:10}  {:5.2f}  {:6.2f}  {:7.2f}  {:3}'.format(file_name, step, e, accept, fail, trials, t3-t2, t4-t3, t5-t4, spg))

        if accept:
            shutil.copy('POSCAR_relax_tmp', 'POSCAR_relax')
            prev_e = e
            POSCAR_text.append(extract_POSCAR_text_lines('POSCAR_init', step, 'init'))
            POSCAR_text.append(extract_POSCAR_text_lines('POSCAR_relax', step, 'relax'))

        if e < best_e:
            best_e = e
            shutil.copy('POSCAR_relax', 'POSCAR_best')

    # 4. Finalize after MC algorithm 
    if 'POSCAR_best' in os.listdir():
        os.rename('POSCAR_best', SAVE_DIR+'/'+file_name)
        with open('Results', 'a') as s:
            s.write('{:20} {:10.4f}\n'.format(file_name, best_e))

    idx = '_'.join(file_name.split('_')[1:])
    with open('XDATCAR_'+idx, 'w') as s:
        for text in POSCAR_text:
            for line in text:
                s.write(line)
