import os, sys, yaml
from topic.topology_csp.basic_tools import *
import re
import math
import numpy as np

import random, datetime
import subprocess
import shutil
import pyrandspg

from topic.topology_csp.poscar import *

########### read input.yaml file #############

input_file = str(sys.argv[1])
with open(input_file, 'r') as f:
    total_yaml = yaml.safe_load(f)

material    = total_yaml['material']
cation_cn   = total_yaml['cation_cn']

# Getting cation information, and sort with number of atoms
cat_info = {k: v for k, v in material.items() if k not in {'Li', 'O'}}
cat_info = sorted(cat_info.items(), key=lambda item: item[1], reverse=True)
# Merging cation information and oxygen information
frame_info = dict(cat_info) | dict([('O', material['O'])])

anion_type = ['O','F','S','Cl']
poscarline = f"{' '.join(frame_info.keys())}\n{' '.join(map(str, frame_info.values()))}"

##############################################

def distance(a,b) :
    return sum([(x-y)**2.0 for x,y in zip(a,b)])**0.5 ;

def dist_pbc(a,b,cell) :
    imagecell = [-1,0,1]
    pbc = [[i,j,k] for i in imagecell for j in imagecell for k in imagecell]
    b_pbc = [[b[i] + cell*pbc[j][i] for i in range(3)] for j in range(len(pbc))]
    return min([distance(a,b_pbc[i]) for i in range(len(pbc))])

at=re.compile('[A-Z][a-z]?')


def make_oxygen():
  pos = read_poscar_dict("POSCAR_cation")
  iteration_limit = len(pos['coor'])-2
 
  # calculate distance info
  pos['coor'] = direct2cartesian(pos['latt'],pos['coor'])
  pos['cartesian'] = True
  distance_array = {}
  for i1,c1 in enumerate(pos['coor']):
   r_data = []

   for i2,c2 in enumerate(pos['coor']):
    if i1 != i2:

      r = calculate_distance(c1,c2,pos['latt'])

      r_data.append([i2,r])

   r_data = sorted(r_data,key=lambda x:x[1]) 

   distance_array[i1] = [i[0] for i in r_data]

  # define n_cation
  n_cation = [i for i in range(len(pos['coor']))]

  # select pairs
  cation_dict = {}

  iteration = 0
  fail = 1

  while fail==1:
  
    pair_lists  = []

    fail=0
    random.shuffle(n_cation)

    for k,a0 in enumerate(pos['coor']):
       
      cation_dict[k] = []

    fail = 0

    # link cations
    for ta in n_cation:

      ta_type = pos['atomarray'][ta]
      
      CN = cation_cn[ta_type]

      # link other cations
      ii = 0

      if len(cation_dict[ta]) < CN:
       while len(cation_dict[ta]) < CN:
        if ii > iteration_limit:
          fail = 1
          break

        candidate = distance_array[ta][ii]  
        candidate_type = pos['atomarray'][candidate]
        candidate_CN   = cation_cn[candidate_type] 

        if candidate not in cation_dict[ta] and len(cation_dict[candidate]) < candidate_CN:

          cation_dict[ta].append(candidate)
          cation_dict[candidate].append(ta)
          pair_lists.append([ta,candidate])

        ii += 1 
    iteration += 1

  # add oxygen

  cation_o_pair = {}

  for n in sorted(n_cation):
    cation_o_pair[n] = []
 
  o_lists = []

  O_num = len(n_cation)
  for pair in pair_lists:

     dmax = 100.0

     a1 = pair[0]
     a2 = pair[1]  

     c1 = pos['coor'][a1]
     c2 = pos['coor'][a2]


     for I in range(3):
      i = float(I-1) 

      for J in range(3):
       j = float(J-1)

       for K in range(3):
        k = float(K-1)

        c2_new = c2 + i*pos['latt'][0] + j*pos['latt'][1] + k*pos['latt'][2]
        d = np.linalg.norm(c2_new-c1)

        if d < dmax:

          dmax = d

          o_atom = (c1+c2_new)/2.0
  
     pos = adatom_dict_fix(pos,o_atom,'O','T')

     cation_o_pair[a1].append(O_num)
     cation_o_pair[a2].append(O_num)
     O_num += 1
 
  write_contcar_dict(pos,"POSCAR")    

  # sort cation_o_pair
  for ckey in cation_o_pair.keys():
    cation_o_pair[ckey] = sorted(cation_o_pair[ckey])

  return cation_o_pair

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


def coo2CONTCAR(filename,output="CONTCAR"):
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

 printBuffer += poscarline 
 printBuffer += '\nCartesian\n'
 for rowCnt in range(numIon):
    for colCnt in range(3):
        printBuffer += '   %12.8f ' %float(Cartesian[rowCnt][colCnt])
    printBuffer += '\n'

 with open(output,"w") as w:
   w.write(printBuffer)

def randspg(composition,elements,volume,tolerance):

  while 1:

    lmin = volume**(1.0/3.0) * 0.4
    lmax = volume**(1.0/3.0) * 2.5

    spg = random.randint(1,230)

    pymin = pyrandspg.LatticeStruct(lmin, lmin, lmin, 60.0, 60.0, 60.0)
    pymax = pyrandspg.LatticeStruct(lmax, lmax, lmax, 120.0, 120.0, 120.0)

    input_ = pyrandspg.RandSpgInput(spg, composition, pymin,pymax, 1.0, volume*0.9, volume*1.2, 100, tolerance, False)

    c = pyrandspg.RandSpg.randSpgCrystal(input_)

    structure = c.getPOSCARString()

    if 'nan' not in structure:
      break

  structure_lists = structure.split('\n')

  with open("POSCAR_cation0","w") as w:

   for i,line in enumerate(structure_lists):

    if i == 0:
     w.write("randspg\n")

    elif i == 5:
     w.write(elements+"\n")

    elif i == 7:
     w.write("Selective dynamics\n")
     w.write(line+'\n')

    elif i > 7 and i < (len(composition)+8):
     w.write(line+" T T T\n")

    elif i >= (len(composition)+8):
     continue

    else:
     w.write(line+"\n")
    
  pos = read_poscar_dict("POSCAR_cation0")
  pos['coor'] = direct2cartesian(pos['latt'],pos['coor']) 

  for i in range(len(pos['coor'])):
    pos['coor'][i][0] += 0.01*random.random()
    pos['coor'][i][1] += 0.01*random.random()
    pos['coor'][i][2] += 0.01*random.random()

  pos['coor'] = cartesian2direct(pos['latt'],pos['coor']) 
  

  write_contcar_dict(pos,"POSCAR_cation")

  return spg
###########
########### input #############
'''
spg_tolerance =[ [[1,1], 3.3],
                 [[1,2], 3.1],
                 [[2,2], 2.8]]
composition = [1 for i in range(16)] + [2 for i in range(8)]
volume      = 800.0
elements = 'Ta P'

randspg(composition,elements,volume,spg_tolerance)
'''
