from topic.topology_csp.basic_tools import read_poscar_dict, calculate_distance


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

            distance = calculate_distance(pos['coor'][c],pos['coor'][a],pos['latt'])
            if distance < bond_cut * scrutinize_factor:
                anion_dict[a].append(c)

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
