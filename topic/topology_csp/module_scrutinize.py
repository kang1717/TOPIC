from topic.topology_csp.basic_tools import read_poscar_dict, calculate_distance


def check_topology(inp, pos):
    cation_cn = inp['cation_cn']
    bond_dict = inp['topology_constraint']
    scrutinize_factor = inp['scrutinize_distance_factor']

    if pos['cartesian'] == False:
        pos['coor'] = np.dot(pos['coor'], pos['latt'])

    # choose cation dict, anion dict
    n_cation = []
    n_anion  = []
    cation_dict = {}
    anion_dict  = {}

    for i,a in enumerate(pos['atomarray']):
        if a not in ['O', 'Li']:
            n_cation.append(i)
            cation_dict[i] = []
        elif a in ['O']:
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

    for a in n_anion:
        atom_a = pos['atomarray'][a]

        for c in n_cation:
            distance = calculate_distance(pos['coor'][c],pos['coor'][a],pos['latt'])
            atom_c = pos['atomarray'][c]

            bond_cut = bond_dict[atom_c+"-"+atom_a]

            if distance < bond_cut * scrutinize_factor:
                anion_dict[a].append(c)

    for anion in anion_dict:
        anion_dict[anion] = tuple(sorted(anion_dict[anion]))

    fail = 0
    for ckey in cation_dict.keys():
        if len(cation_dict[ckey]) != cation_cn[pos['atomarray'][ckey]]:
            fail += 1
    for akey in anion_dict.keys():
        if len(anion_dict[akey]) != 2:
            fail += 1

    # If do not satisfy cornershaing, return 999
    if len(set(list(anion_dict.values()))) != int(inp['material']['O']):
        return 999

    return fail

def scrutinize(pos, cation_ref, total_yaml):
    cation_cn = total_yaml['cation_cn']
    bond_dict = total_yaml['distance_constraint']
    scrutinize_factor = total_yaml['scrutinize_distance_factor']

    #pos = read_poscar_dict(contcarname)
    # choose cation dict, anion dict
    n_cation = []
    n_anion  = []
    cation_dict = {}
    anion_dict  = {}

    for i,a in enumerate(pos['atomarray']):
        if a not in ['O']:
            n_cation.append(i)
            cation_dict[i] = []
        else:
            n_anion.append(i)
            anion_dict[i] = []


    # write a cation_dict of its own
    for c in n_cation:
        atom_c = pos['atomarray'][c]

        for a in cation_ref[c]:
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
