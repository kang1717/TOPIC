from topic.topology_csp.basic_tools import direct2cartesian, calculate_distance, \
                                    adatom_dict_fix, cartesian2direct
import os, sys, yaml
import numpy as np
import random
import pyrandspg
import spglib


atomic_mass = dict(H=1.01, He=4.00, Li=6.94, Be=9.01, B=10.81, C=12.01,
                   N=14.01, O=16.00, F=19.00, Ne=20.18, Na=22.99, Mg=24.31,
                   Al=26.98, Si=28.09, P=30.97, S=32.07, Cl=35.45,
                   K=39.10, Ca=40.07, Sc=44.96, Ti=47.87, V=50.94, Cr=52.00,
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


def generate_initial_structure_random(total_yaml):
    trial = 0
    done = 0
    while done == 0:
        if total_yaml['spg_seed'] == None:
            spg = random.randint(1, 230)
        else:
            spg = total_yaml['spg_seed']

        # Generate cation sites
        pos_cat = generate_cation_sites(total_yaml, spg) # Cartesian coordinates
        if pos_cat == None:
            continue

        # Generate oxygen sites which not have P1 symmetry
        neighbor_array = calculate_distance_o_sites(pos_cat)
        for spg_trial in range(1000):
            trial += 1
            pos, bond_dict = generate_o_sites(pos_cat, total_yaml, neighbor_array)
            spg0 = get_space_group(pos) 
            if spg0 not in [0, 1]:
                done = 1
                break

    return pos, bond_dict, trial, spg, spg0

def generate_initial_structure_shortest(total_yaml):
    trial = 0
    done = 0
    while done == 0:
        if total_yaml['spg_seed'] == None:
            spg = random.randint(1, 230)
        else:
            spg = total_yaml['spg_seed']

        # Generate cation sites
        for spg_trial in range(100):
            trial += 1
            pos_cat = generate_cation_sites(total_yaml, spg) # Cartesian coordinates
            if pos_cat == None:
                break

            pos, bond_dict = make_oxygen_shortest_bond(pos_cat, total_yaml)
            if pos != None:
                done = 1
                break

    spg0 = get_space_group(pos) 

    return pos, bond_dict, trial, spg, spg0

def make_oxygen_shortest_bond(pos, total_yaml):
    cation_cn  = total_yaml['cation_cn']
    num_o = int(total_yaml['material']['O'])

    if pos['cartesian'] == True:
        direct_coord = cartesian2direct(pos['latt'], pos['coor'])
    else:
        direct_coord = pos['coor']
        pos['coor'] = direct2cartesian(pos['latt'], pos['coor'])
        pos['cartesian'] = True

    cn_num_dict = dict()
    bond_lengths = []
    for i1, c1 in enumerate(pos['coor']):
        cn_num_dict[i1] = 0
        for i2, c2 in enumerate(pos['coor']):
            if i1 < i2:
                for I in range(-1,2):
                    for J in range(-1,2):
                        for K in range(-1,2):
                            i=  float(I); j = float(J); k = float(K)
                            c1_new = c1 + pos['latt'][0]*i + pos['latt'][1]*j + pos['latt'][2]*k
                            dist = np.linalg.norm(c1_new-c2)
                            o_pos = (c1_new+c2)/2.0
                            bond_lengths.append(((i1, i2), dist, o_pos))


    # Step 3: Sort bond lengths
    cation_o_pair = {}
    for n in range(len(pos['coor'])):
        cation_o_pair[n] = []

    bond_lengths_sorted = sorted(bond_lengths, key=lambda x: x[1])

    # Check if CN condition satisfy
    for bond_info, _, _ in bond_lengths_sorted[:num_o]:
        i, j = bond_info
        cn_num_dict[i] += 1
        cn_num_dict[j] += 1

    no_cs = 0
    for i in range(len(pos['coor'])):
        cat_type = pos['atomarray'][i]
        cat_CN = cation_cn[cat_type]
        if cn_num_dict[i] != cat_CN:
            no_cs = 1
            break
    if no_cs == 1:
        return None, None

    O_idx = len(pos['coor'])
    for bond_info, length, o_pos in bond_lengths_sorted[:num_o]:
        i, j = bond_info

        # Add an oxygen atom at the midpoint
        pos = adatom_dict_fix(pos, o_pos, 'O', 'T')
        cation_o_pair[i].append(O_idx)
        cation_o_pair[j].append(O_idx)
        O_idx += 1

    # sort cation_o_pair
    for ckey in cation_o_pair.keys():
        cation_o_pair[ckey] = sorted(cation_o_pair[ckey])

    return pos, cation_o_pair

def generate_cation_sites(total_yaml, spg):
    ######### input parameters ########
    material = total_yaml['material']
    cat_info = {k: v for k, v in material.items() if k not in {'Li', 'O'}}
    cat_info = sorted(cat_info.items(), key=lambda item: item[1], reverse=True)
    composition = []
    for i in range(len(cat_info)):
        composition += [i+1 for _ in range(cat_info[i][1])]
    elements = ' '.join([item[0] for item in cat_info])

    volume = total_yaml['volume']

    # distance constraint
    constraint_dict = total_yaml['generation_constraint']
    gen_factor = total_yaml['generation_distance_factor']
    distance_tolerance = []
    for i in range(len(cat_info)):
        for j in range(i, len(cat_info)):
            if f"{cat_info[i][0]}-{cat_info[j][0]}" in constraint_dict:
                constraint = constraint_dict[f"{cat_info[i][0]}-{cat_info[j][0]}"]
            else:
                constraint = constraint_dict[f"{cat_info[j][0]}-{cat_info[i][0]}"]
            distance_tolerance.append([[i+1,j+1], constraint*gen_factor])
    ####################################

    # Generate random cation sites
    lmin = volume**(1.0/3.0) * 0.4
    lmax = volume**(1.0/3.0) * 2.5
    pymin = pyrandspg.LatticeStruct(lmin, lmin, lmin, 60.0, 60.0, 60.0)
    pymax = pyrandspg.LatticeStruct(lmax, lmax, lmax, 120.0, 120.0, 120.0)
    input_ = pyrandspg.RandSpgInput(spg, composition, pymin, pymax, 1.0, \
                            volume*0.9, volume*1.2, 100, distance_tolerance, False)

    c = pyrandspg.RandSpg.randSpgCrystal(input_)
    structure = c.getPOSCARString()
    if 'nan' in structure:
        return None

    pos = transform_poscar_text_to_pos_dict(structure, total_yaml)
    return pos

def transform_poscar_text_to_pos_dict(poscar_text, total_yaml):
    material = total_yaml['material']
    cat_info = {k: v for k, v in material.items() if k not in {'Li', 'O'}}
    cat_info = sorted(cat_info.items(), key=lambda item: item[1], reverse=True)
    N = sum([cat_info[i][1] for i in range(len(cat_info))])

    structure_lists = poscar_text.split('\n')

    pos = dict()
    matrix_list = [list(map(float, line.split())) for line in structure_lists[2:5]]
    pos['latt'] = np.array(matrix_list)

    coords_list = []
    for line in structure_lists[8:N+8]:
        coords_list.append(list(map(float, line.split())))
    pos['coor'] = np.array(coords_list)
    pos['coor'] = direct2cartesian(pos['latt'], pos['coor'])

    atomlist = [a[0] for a in cat_info]
    numlist = [a[1] for a in cat_info]
    atomarray = [atom for atom,count in zip(atomlist,numlist) for _ in range(count)]
    numarray = [j for j in range(len(atomlist)) for _ in range(numlist[j])]

    pos['cartesian'] = True
    pos['atomlist'] = atomlist
    pos['numlist'] = numlist 
    pos['atomarray'] = np.array(atomarray)
    pos['numarray'] = np.array(numarray)
    pos['fix'] = np.array([['T', 'T', 'T'] for i in range(len(pos['atomarray']))])

    return pos

def generate_o_sites(pos, total_yaml, neighbor_array):
    # Link two cations and make pair_lists
    bond_dict = make_bond_dict_in_random_order(pos, total_yaml, neighbor_array)

    # Place oxygen atom in between two cations
    cation_o_pair = {}
    for n in range(len(pos['coor'])):
        cation_o_pair[n] = []

    pair_lists = [] # duplicate list
    dup_o_pos = [] # duplicate list
    O_idx = len(pos['coor'])
    for cat_idx in bond_dict.keys():
        for neighbor in bond_dict[cat_idx]:
            neighbor_idx = neighbor[0]
            if (neighbor_idx, cat_idx) in pair_lists:
                if tuple(neighbor[1]) in dup_o_pos:
                    continue
                else:
                    dup_o_pos.append(tuple(neighbor[1]))
            else:
                pair_lists.append((cat_idx, neighbor_idx))
                dup_o_pos.append(tuple(neighbor[1]))

            #pos = adatom_dict_fix(pos, pos_o[f'{cat_idx}-{neighbor_idx}'], 'O', 'T')
            pos = adatom_dict_fix(pos, neighbor[1], 'O', 'T')
            cation_o_pair[cat_idx].append(O_idx)
            cation_o_pair[neighbor_idx].append(O_idx)
            O_idx += 1

    # sort cation_o_pair
    for ckey in cation_o_pair.keys():
        cation_o_pair[ckey] = sorted(cation_o_pair[ckey])

    return pos, cation_o_pair

def make_bond_dict_in_random_order(pos, total_yaml, neighbor_array):
    cation_cn  = total_yaml['cation_cn']
    n_cation = [i for i in range(len(pos['coor']))]

    trial = 0
    while True:
        trial += 1
        bond_dict = {}
        for k, a0 in enumerate(pos['coor']):
            bond_dict[k] = []

        random.shuffle(n_cation)
        success = True
        for cat_idx in n_cation:
            cat_type = pos['atomarray'][cat_idx]
            cat_CN = cation_cn[cat_type]
            if len(bond_dict[cat_idx]) >= cat_CN: # Check center atom CN
                continue

            for neighbor in neighbor_array[cat_idx]:
                neighbor_idx = neighbor[0]
                neighbor_type = pos['atomarray'][neighbor_idx]
                neighbor_CN   = cation_cn[neighbor_type]
                if len(bond_dict[neighbor_idx]) >= neighbor_CN: # Check neighbor CN
                    continue

                #if neighbor_idx not in bond_dict[cat_idx]:
                #    bond_dict[cat_idx].append([neighbor_idx, neighbor[2]])
                #    bond_dict[neighbor_idx].append([cat_idx, neighbor[2]])
                bond_dict[cat_idx].append([neighbor_idx, neighbor[2]])
                bond_dict[neighbor_idx].append([cat_idx, neighbor[2]])

                if len(bond_dict[cat_idx]) >= cat_CN:
                    break

            if len(bond_dict[cat_idx]) < cat_CN:
                success = False
                break

        if success:
            return bond_dict


def get_space_group(pos):
    lattice = pos['latt']
    if pos['cartesian'] == True:
        positions = cartesian2direct(pos['latt'], pos['coor'])
    else:
        positions = pos['coor']

    numbers = pos['numarray']

    cell = (lattice, positions, numbers)

    spacegroup = spglib.get_symmetry_dataset(cell, symprec=1)
    if spacegroup == None:
        return 0
    else:
        return spacegroup.number

########################### backup ###########################
def coo2CONTCAR(filename, output="CONTCAR"):
    #------------------------- simple structure file part -----------------------#
    LammpsLines = open(filename).readlines()
    LatticeVector = [[0,0,0],[0,0,0],[0,0,0]]
    name = LammpsLines[0]
    #if '#' == LammpsLines[0][0]:
    #    name = LammpsLines[0].split('#',1)[1]
    for cnt in range(1,len(LammpsLines)):
        if 'atoms' in LammpsLines[cnt]:
            numIon = int(LammpsLines[cnt].split()[0])
        elif 'atom types' in LammpsLines[cnt]:
            numElements = int(LammpsLines[cnt].split()[0])
        elif 'xlo' in LammpsLines[cnt]:
            LatticeVector[0][0] = float(LammpsLines[cnt].split()[1]) - float(LammpsLines[cnt].split()[0])
        elif 'ylo' in LammpsLines[cnt]:
            LatticeVector[1][1] = float(LammpsLines[cnt].split()[1]) - float(LammpsLines[cnt].split()[0])
        elif 'zlo' in LammpsLines[cnt]:
            LatticeVector[2][2] = float(LammpsLines[cnt].split()[1]) - float(LammpsLines[cnt].split()[0])
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
    MassesLines.sort(key=lambda x: int(x.split()[0]))
    #AtomsLines.sort(key=lambda x: float(x.split()[4]))   # sort about z coordinate
    #AtomsLines.sort(key=lambda x: int(x.split()[1]))
    ElementList = []
    AMItems = list(atomic_mass.items())
    for line in MassesLines:
        mass = round(float(line.split()[1]),0)
        for Tup in AMItems:
            if round(Tup[1],0) == mass:
                ElementList.append(Tup[0])
                break
    NumIonList = [0]*numElements
    Cartesian = []
    for line in AtomsLines:
        NumIonList[int(line.split()[1])-1] += 1
        Cartesian.append(line.split()[2:])
    #------------------------------------------------------------------#

    #------------------------ print scripts -----------------------------#
    printBuffer = ''
    printBuffer += '(Converted from LAMMPS) '+name+' 1.0\n'
    for rowCnt in range(3):
        for colCnt in range(3):
            printBuffer += '   %12.8f ' %float(LatticeVector[rowCnt][colCnt])
        printBuffer += '\n'
    #for comp in ElementList:
    #   printBuffer += '  %s' %comp
    #printBuffer += '\n'
    #for comp in NumIonList:
    #    printBuffer += '  %s' %comp

    printBuffer += f"{' '.join(frame_info.keys())}\n{' '.join(map(str, frame_info.values()))}"
    printBuffer += '\nCartesian\n'
    for rowCnt in range(numIon):
        for colCnt in range(3):
            printBuffer += '   %12.8f ' %float(Cartesian[rowCnt][colCnt])
        printBuffer += '\n'

    with open(output,"w") as w:
        w.write(printBuffer)


def randspg_all(composition, elements, volume, tolerance, spg):
    lmin = volume**(1.0/3.0) * 0.4
    lmax = volume**(1.0/3.0) * 2.5

    pymin = pyrandspg.LatticeStruct(lmin, lmin, lmin, 60.0, 60.0, 60.0)
    pymax = pyrandspg.LatticeStruct(lmax, lmax, lmax, 120.0, 120.0, 120.0)

    input_ = pyrandspg.RandSpgInput(spg, composition, pymin, pymax, 1.0, volume*0.9, volume*1.2, 100, tolerance, False)

    c = pyrandspg.RandSpg.randSpgCrystal(input_)

    structure = c.getPOSCARString()

    if 'nan' in structure:
        return None

    structure_lists = structure.split('\n')

    pos = dict()
    matrix_list = [list(map(float, line.split())) for line in structure_lists[2:5]]
    pos['latt'] = np.array(matrix_list)

    coords_list = [list(map(float, line.split())) for line in structure_lists[8:len(composition)+8]]
    pos['coor'] = np.array(coords_list)

    pos['atomlist'] = [a[0] for a in host_info]
    pos['numlist'] = [a[1] for a in host_info]
    pos['cartesian'] = False

    pos['atomarray'] = np.array([atom for atom, count in zip(pos['atomlist'], pos['numlist']) for _ in range(count)])
    pos['numarray'] = np.array([j for j in range(len(pos['atomlist'])) for _ in range(pos['numlist'][j])])
    pos['fix'] = np.array([['T', 'T', 'T'] for i in range(len(pos['atomarray']))])

    return pos

def read_structure(file_path):
    structure = Structure.from_file(file_path)
    new_sites = []
    for site in structure:
        if site.species_string != 'O' and site.species_string != 'Li':
            new_sites.append(site)

    new_structure = Structure(
            lattice=structure.lattice,
            species=[site.species_string for site in new_sites],
            coords=[site.frac_coords for site in new_sites],
            coords_are_cartesian=False
            )
    return new_structure

def calculate_distance_o_sites(pos_cat):
    distance_array = {}
    for i1, c1 in enumerate(pos_cat['coor']):
        r_data = []
        for i2, c2 in enumerate(pos_cat['coor']):
            if i1 != i2:
                #r = calculate_distance(c1, c2, pos_cat['latt'])
                r = 10000.0
                for I in range(-1,2):
                    for J in range(-1,2):
                        for K in range(-1,2):
                            i=  float(I); j = float(J); k = float(K)
                            c1_new = c1 + pos_cat['latt'][0]*i + pos_cat['latt'][1]*j + pos_cat['latt'][2]*k

                            dist = np.linalg.norm(c1_new-c2)
                            o_pos = (c1_new+c2)/2.0
                            r_data.append([i2, dist, o_pos])

                            #temp = np.linalg.norm(c1_new-c2)
                            #if temp < r:
                            #    r = temp
                            #    o_pos_dict[f'{i1}-{i2}'] = (c1_new+c2)/2.0
                            #    o_pos_dict[f'{i2}-{i1}'] = (c1_new+c2)/2.0

                #r_data.append([i2, r])

        r_data = sorted(r_data, key=lambda x:x[1])
        distance_array[i1] = r_data
        #distance_array[i1] = [i[0] for i in r_data]

    return distance_array

def check_oo_distance(pos, inp):
    fail = 0

    cation_cn = inp['cation_cn']
    bond_dict = inp['distance_constraint']
    gen_factor = inp['generation_distance_factor']

    if pos['cartesian'] == False:
        pos['coor'] = np.dot(pos['coor'], pos['latt'])

    for a1 in range(len(pos['atomarray'])):
        if pos['atomarray'][a1] != 'O':
            continue
        for a2 in range(len(pos['atomarray'])):
            if a2 <= a1:
                continue
            if pos['atomarray'][a2] != 'O':
                continue

            distance = calculate_distance(pos['coor'][a1],pos['coor'][a2],pos['latt'])
            bond_cut = bond_dict["O-O"]

            if distance < bond_cut * gen_factor:
                return 1

    return fail
