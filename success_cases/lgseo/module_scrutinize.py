import sys
sys.path.append("/home/sung.w.kang/utill")
from basic_tools import *
import copy
from copy import deepcopy
import random


# bond info
bond_dict  = {}
bond_dict['Ga-O'] = 1.94
bond_dict['Se-O'] = 1.76
bond_dict['O-Ga'] = 1.94
bond_dict['O-Se'] = 1.76

anion_type = ['O','F','S','Cl']

cation_cn  = {'Ga':6, 'Se':3}


def scrutinize(contcarname,cation_ref):
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

    if distance < bond_cut*1.1:

      cation_dict[c].append(a)


 for cation in cation_dict:
  cation_dict[cation] = sorted(cation_dict[cation])


 fail = 0
 for ckey in cation_dict.keys():

  if len(cation_dict[ckey]) != cation_cn[pos['atomarray'][ckey]]:

    fail += 1

 return fail

