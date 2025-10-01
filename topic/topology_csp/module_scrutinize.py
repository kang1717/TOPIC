from topic.topology_csp.basic_tools import read_poscar_dict, calculate_distance
import numpy as np


def check_topology(inp, pos):
    if pos['cartesian'] == False:
        pos['coor'] = np.dot(pos['coor'], pos['latt'])

    # If topology is not corner sharing, fail = 999
    # Else, fail contains # of atoms that do not satisfying coordinations
    fail = check_anion_CN(pos, inp)
    if fail != 999:
        fail += check_cation_CN(pos, inp)

    return fail

def check_cation_CN(pos, inp):
    cation_cn = inp['cation_cn']
    bond_dict = inp['topology_constraint']
    scrutinize_factor = inp['scrutinize_distance_factor']

    cation_list = []
    anion_list = []
    for i,a in enumerate(pos['atomarray']):
        if a not in ['O', 'Li']:
            cation_list.append(i)
        elif a in ['O']:
            anion_list.append(i)

    cation_dict = {}
    for c in cation_list:
        cation_dict[c] = []
        atom_c = pos['atomarray'][c]

        for a in anion_list:
            atom_a = pos['atomarray'][a]
            bond_cut = bond_dict[atom_c+"-"+atom_a]

            distance = calculate_distance(pos['coor'][c],pos['coor'][a],pos['latt'])
            if distance < bond_cut * scrutinize_factor:
                cation_dict[c].append(a)

    fail = 0
    for ckey in cation_dict.keys():
        if len(cation_dict[ckey]) != cation_cn[pos['atomarray'][ckey]]:
            fail += 1

    return fail

def check_anion_CN(pos, inp):
    cation_cn = inp['cation_cn']
    bond_dict = inp['topology_constraint']
    scrutinize_factor = inp['scrutinize_distance_factor']

    cation_list = []
    anion_list  = []
    for i,a in enumerate(pos['atomarray']):
        if a not in ['O', 'Li']:
            cation_list.append(i)
        elif a in ['O']:
            anion_list.append(i)

    anion_dict  = {}
    for a in anion_list:
        anion_dict[a] = []
        atom_a = pos['atomarray'][a]

        for c in cation_list:
            atom_c = pos['atomarray'][c]
            bond_cut = bond_dict[atom_c+"-"+atom_a]

            #distance = calculate_distance(pos['coor'][c],pos['coor'][a],pos['latt'])
            distance = 10000.0
            for I in range(-1,2):
                for J in range(-1,2):
                    for K in range(-1,2):
                        i=  float(I); j = float(J); k = float(K)
                        c1_new = pos['coor'][c] + pos['latt'][0]*i + pos['latt'][1]*j + pos['latt'][2]*k
                        dist = np.linalg.norm(c1_new-pos['coor'][a])
                        if dist < distance:
                            distance = dist
                            image = (i, j ,k)

            if distance < bond_cut * scrutinize_factor:
                anion_dict[a].append((c, image))

    for anion in anion_dict:
        anion_dict[anion] = tuple(sorted(anion_dict[anion]))

    # If do not satisfy cornershaing, return 999
    if len(set(list(anion_dict.values()))) != int(inp['material']['O']):
        return 999

    fail = 0
    for akey in anion_dict.keys():
        if len(anion_dict[akey]) != 2:
            fail += 1

    return fail
