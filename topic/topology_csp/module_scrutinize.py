import sys, yaml
from topic.topology_csp.basic_tools import read_poscar_dict, calculate_distance
import copy
from copy import deepcopy
import random

########### read input.yaml file #############

#input_file = str(sys.argv[1])
#with open(input_file, 'r') as f:
#    total_yaml = yaml.safe_load(f)

anion_type = ['O','F','S','Cl']

##############################################

def scrutinize(contcarname, cation_ref, total_yaml):
    cation_cn = total_yaml['cation_cn']
    bond_dict = total_yaml['distance_constraint']

    pos = read_poscar_dict(contcarname)
    # choose cation dict, anion dict
    n_cation = []
    n_anion  = []
    cation_dict = {}
    anion_dict  = {}

    for i,a in enumerate(pos['atomarray']):
        if a not in anion_type:
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

            if distance < bond_cut*1.15:
                cation_dict[c].append(a)

    for cation in cation_dict:
        cation_dict[cation] = sorted(cation_dict[cation])

    fail = 0
    for ckey in cation_dict.keys():
        if len(cation_dict[ckey]) != cation_cn[pos['atomarray'][ckey]]:
            fail += 1

    return fail
