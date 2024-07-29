import os,sys
sys.path.append("/home/sung.w.kang/utill")
from basic_tools import *
import re
import math
import numpy as np

import os,sys,random,datetime
import subprocess
import shutil
import pyrandspg

import time
from time import time
from poscar import *

anion_type = ['O','F','S','Cl']

cation_cn  = {'Ti':6, 'P':4}

def distance(a,b) :
    return sum([(x-y)**2.0 for x,y in zip(a,b)])**0.5 ;

def dist_pbc(a,b,cell) :
    imagecell = [-1,0,1]
    pbc = [[i,j,k] for i in imagecell for j in imagecell for k in imagecell]
    b_pbc = [[b[i] + cell*pbc[j][i] for i in range(3)] for j in range(len(pbc))]
    return min([distance(a,b_pbc[i]) for i in range(len(pbc))])

at=re.compile('[A-Z][a-z]?')

ion_radii_table = {"Zn1":0.74,"Zn2":0.88,"Pb4":0.915,"Ba2":1.49,"Pb2":1.33,"Ca2":1.14,"S-2":1.7,"F-1":1.19,"Ta4":0.82,"Ta5":0.78,"Cl7":0.41,"Ta3":0.86,"O-2":1.26,"Ge2":0.87,"Mo3":0.83,"F7":0.22,"Ga3":0.76,"Mo6":0.73,"Mo5":0.75,"Mo4":0.79,"Mg2":0.86,"W5":0.76,"Cl-1":1.67,"B3":0.41,"Bi5":0.9,"Bi3":1.17,"Tl3":1.025,"Tl1":1.64,"Pd4":0.755,"Pd3":0.9,"Pd2":1.0,"Pd1":0.73,"Sr2":1.32,"Ag3":0.89,"Ag2":1.08,"Ag1":1.29,"Os8":0.53,"W6":0.74,"Tc7":0.7,"Tc4":0.785,"Tc5":0.74,"Os5":0.715,"Os4":0.77,"Os7":0.665,"Os6":0.685,"W4":0.8,"Pt5":0.71,"Pt4":0.765,"Pt2":0.94,"Sc3":0.885,"P3":0.58,"Lu3":1.001,"Te-2":2.07,"P5":0.52,"Hg2":1.16,"Re6":0.69,"Re7":0.67,"Re4":0.77,"Re5":0.72,"Hg1":1.33,"Cs1":1.81,"Sb3":0.9,"Sb5":0.74,"H1":-0.04,"Ru8":0.5,"Ru3":0.82,"Ru7":0.52,"Ru4":0.76,"Ru5":0.705,"S6":0.43,"Rb1":1.66,"S4":0.51,"K1":1.52,"Be2":0.59,"Nb4":0.82,"Se6":0.56,"Se4":0.64,"Nb3":0.86,"Nb5":0.78,"In3":0.94,"Te4":1.11,"Te6":0.7,"C4":0.3,"Au1":1.51,"Au3":0.99,"Au5":0.71,"Cl5":0.26,"N3":0.3,"N5":0.27,"I-1":2.06,"Br-1":1.82,"Ge4":0.67,"Hf4":0.85,"I5":1.09,"N-3":1.32,"Br7":0.53,"Br5":0.45,"Br3":0.73,"I7":0.67,"Cd2":1.09,"Xe8":0.62,"Al3":0.675,"Zr4":0.86,"Si4":0.54,"Ti4":0.745,"Ir3":0.82,"Ir4":0.765,"Ir5":0.71,"Ti2":1.0,"Ti3":0.81,"Na1":1.16,"Li1":0.9,"Se-2":1.84,"As3":0.72,"As5":0.6,"Rh3":0.805,"Rh5":0.69,"Rh4":0.74,"Sn4":0.83,"Y3":1.04}



non_metal = ['H','O','F','S','Cl',
		'Se','Br','Te','I','N','C','Si','P','Ge','As','Sn','Sb']


zero = ['He','Ne','Ar','Xe']
one = ['Li','Na','K','Rb','Cs']
two = ['Be','Mg','Ca','Sr','Ba']
three = ['B','Al','Ga','In','Sc','Y']

oxidation = {}

oxidation['H'] = [1,-1]
oxidation['D'] = [1,-1]

oxidation['C'] = [4,3,2,1,0,-1,-2,-3,-4]
oxidation['N'] = [3, -3, 5]
oxidation['O'] = [-2]
oxidation['F'] = [-1]

oxidation['Si'] = [4, -4]
oxidation['P'] = [5, 3, -3]
oxidation['S'] = [6, -2, 2, 4]
oxidation['Cl'] = [-1, 1, 3, 5, 7]

oxidation['Ti'] = [4]
oxidation['Ge'] = [2, 4, -4]
oxidation['As'] = [3, -3, 5]
oxidation['Se'] = [4, 6, 2, -2]
oxidation['Br'] = [-1, 1, 3, 5]
oxidation['Kr'] = [0, 2]

oxidation['Zn'] = [1,2]
oxidation['Zr'] = [4]
oxidation['Nb'] = [5]
oxidation['Mo'] = [6, 4]
oxidation['Tc'] = [7, 4]
oxidation['Ru'] = [4, 3]
oxidation['Rh'] = [3]
oxidation['Pd'] = [2, 4, 0]
oxidation['Ag'] = [1]
oxidation['Cd'] = [2]
oxidation['Sn'] = [4, 2, -4]
oxidation['Sb'] = [3, 5, -3]
oxidation['Te'] = [4, 6, 2, -2]
oxidation['I'] = [-1, 1,3, 5, 7]


oxidation['Hf'] = [4]
oxidation['Ta'] = [5]
oxidation['W'] = [6, 4]
oxidation['Re'] = [4]
oxidation['Os'] = [4]
oxidation['Ir'] = [3, 4]
oxidation['Pt'] = [4, 2]
oxidation['Au'] = [3, 1]
oxidation['Hg'] = [2, 1]
oxidation['Tl'] = [1, 3]
oxidation['Pb'] = [2, 4]
oxidation['Bi'] = [3]

oxidation['Lu'] = [3]
oxidation['La'] = [3]

for i in zero:
	oxidation[i] = [0]
for i in one:
	oxidation[i] = [1]
for i in two:
	oxidation[i] = [2]

for i in three:
	oxidation[i] = [3]


ion_radii_table['Li0'] = 145.*0.01
ion_radii_table['Be0'] = 105.*0.01
ion_radii_table['B0'] =  85.*0.01
ion_radii_table['Na0'] = 180.*0.01
ion_radii_table['Mg0'] = 150.*0.01
ion_radii_table['Al0'] = 125.*0.01
ion_radii_table['K0'] = 220.*0.01
ion_radii_table['Ca0'] = 180.*0.01
ion_radii_table['Sc0'] = 160.*0.01
ion_radii_table['Ti0'] = 140.*0.01
ion_radii_table['Ga0'] = 130.*0.01
ion_radii_table['Ge0'] = 125.*0.01
ion_radii_table['Rb0'] = 235.*0.01
ion_radii_table['Sr0'] = 200.*0.01
ion_radii_table['Y0'] = 180.*0.01
ion_radii_table['Zr0'] = 155.*0.01
ion_radii_table['Nb0'] = 145.*0.01
ion_radii_table['Mo0'] = 145.*0.01
ion_radii_table['Ru0'] = 130.*0.01
ion_radii_table['Rh0'] = 135.*0.01
ion_radii_table['Pd0'] = 140.*0.01
ion_radii_table['Ag0'] = 160.*0.01
ion_radii_table['Cd0'] = 155.*0.01
ion_radii_table['In0'] = 155.*0.01
ion_radii_table['Sn0'] = 145.*0.01
ion_radii_table['Sb0'] = 145.*0.01
ion_radii_table['Cs0'] = 260.*0.01
ion_radii_table['Ba0'] = 215.*0.01
ion_radii_table['Lu0'] = 227.*0.01
ion_radii_table['Hf0'] = 155.*0.01
ion_radii_table['Ta0'] = 145.*0.01
ion_radii_table['W0'] = 135.*0.01
ion_radii_table['Re0'] = 135.*0.01
ion_radii_table['Os0'] = 130.*0.01
ion_radii_table['Ir0'] = 135.*0.01
ion_radii_table['Pt0'] = 135.*0.01
ion_radii_table['Au0'] = 135.*0.01
ion_radii_table['Hg0'] = 150.*0.01
ion_radii_table['Tl0'] = 190.*0.01
ion_radii_table['Pb0'] = 180.*0.01
ion_radii_table['Bi0'] = 160.*0.01
ion_radii_table['La0'] = 195.*0.01


atom_vol = {'H':5.08,'Li':22.6,'Be':36,'B':13.24,'C':13.87,'N':11.8,'O':11.39,'F':11.17,'Na':26,'Mg':36,'Al':39.6,'Si':37.3,'P':29.5,'S':25.2,'Cl':25.8,'K':36,'Ca':45,'Sc':42,'Ti':27.3,'V':24,'Cr':28.1,'Mn':31.9,'Fe':30.4,'Co':29.4,'Ni':26,'Cu':26.9,'Zn':39,'Ga':37.8,'Ge':41.6,'As':36.4,'Se':30.3,'Br':32.7,'Rb':42,'Sr':47,'Y':44,'Zr':27,'Nb':37,'Mo':38,'Tc':38,'Ru':37.3,'Rh':31.2,'Pd':35,'Ag':35,'Cd':51,'In':55,'Sn':52.8,'Sb':48,'Te':46.7,'I':46.2,'Xe':45,'Cs':46,'Ba':66,'La':58,'Ce':54,'Pr':57,'Nd':50,'Sm':50,'Eu':53,'Gd':56,'Tb':45,'Dy':50,'Ho':42,'Er':54,'Tm':49,'Yb':59,'Lu':35,'Hf':40,'Ta':43,'W':38.8,'Re':42.7,'Os':41.9,'Ir':34.3,'Pt':38,'Au':43,'Hg':38,'Tl':54,'Pb':52,'Bi':60,'Ac':74,'Th':56,'Pa':60,'U':58,'Np':45,'Am':17}




def make_structure(inp,V):

 at=re.compile('[A-Z][a-z]?')
 elements=at.findall(inp)
 number = re.findall("\d+", inp)
 float_number = [float(i) for i in number]
 int_number = [int(i) for i in number]




 index = [0]*len(elements)
 reference = [len(oxidation[i]) for i in elements]


 tagtag = False
 for kk in non_metal :
   for j,jtem in enumerate(elements):
    if jtem in non_metal :
      tagtag = True
 oxi_tmp = []
 if tagtag :
   tag = True
   while tag:
    oxi_state = [oxidation[elements[i]][index[i]] for i in range(len(elements))]
    charge = 0
    for i in range(len(elements)):
     charge += oxi_state[i]*int_number[i]

    if charge == 0:
     tag = False
     for i,item in enumerate(elements):
      oxi_tmp.append(oxi_state[i])
    else:
      index[-1] += 1
      index_valid = [index[i] < reference[i] for i in range(len(index))]
      while False in index_valid:
       point = index_valid.index(False)
       index[point-1] += 1
       for i in range(len(index)-point):
        index[point+i] = 0
       index_valid = [index[i] < reference[i] for i in range(len(index))]
       if False in index_valid:
        if index_valid.index(False) == 0:
         tag = False
         break
 else:
  for i,item in enumerate(elements):
   oxi_tmp.append(0)

 str_oxi_tmp = [str(i) for i in oxi_tmp]

 if len(oxi_tmp) == 0 :
	
   vol = 0
   atom_num = 0

   for i,item in enumerate(elements):
    vol += atom_vol[item]*float_number[i]
    atom_num += float_number[i]
	
    atom_vol0 = (vol/atom_num) / 1.391 

 else:
  if elements[0]+str_oxi_tmp[0] in ion_radii_table and elements[1]+str_oxi_tmp[1] in ion_radii_table and elements[2]+str_oxi_tmp[2] in ion_radii_table :
   ion_vol = []
   vol = 0
   atom_num = 0
   for i,item in enumerate(elements):
    vol += (4./3.)*math.pi*float_number[i]*(ion_radii_table[item+str_oxi_tmp[i]]**3)
    atom_num += float_number[i]
   vol = vol/sum(float_number)
   atom_vol0 = vol/0.6654
	
  else:
   vol = 0
   atom_num = 0

   for i,item in enumerate(elements):
    vol += atom_vol[item]*float_number[i]
    atom_num += float_number[i]
		
    atom_vol0 = (vol/atom_num) / 1.391

 ##########################################
 # python spray.py output
 ##########################################
 POSCAR = open('POSCAR_cation','w') ;
 potdir = '/home/sung.w.kang/potcar_gga_paw_pbe54/'     # POTCAR location ; read atom mass using POTCAR

 ##########################################
 # Inputs about amorphous
 ##########################################
 density = V

 atomname = []
 for atomi in elements:
  atomname.append(atomi)
 natom    = int_number;

 for i in range(len(atomname)):
    atomi = atomname[i]
    if os.path.isdir(potdir+atomi) == False:
        if os.path.isdir(potdir+atomi+"_pv"):
            atomname[i] = atomname[i]+"_pv"

        elif os.path.isdir(potdir+atomi+"_sv"):
            atomname[i] = atomname[i]+"_sv"

        else:
            a=1

 nelement = len(atomname) ;
 totatom = sum(natom) ;
 ##########################################
 # atomic cutoff radius
 ##########################################
 cutoff = [[3.163, 3.37],
          [3.37 , 2.8] ]

 for i in range(2):
  for j in range(2):
   cutoff[i][j] *= 0.7
 
##########################################
 # Preallocate & grep MASS from POTCAR
 ##########################################
 exatom_position = [[0 for i in range(3)] for j in range(totatom)];
 atommass = [0.0 for i in range(nelement)] ;
 for i in range(nelement):
  command = ''.join(['grep MASS ', potdir, '/', atomname[i], '/POTCAR | awk \'{print $3}\' | cut -d";" -f1'])
  atommass[i] = float(subprocess.getoutput(command))

 #cellpar = (sum([a*b for a,b in zip(atommass,map(float,natom))])*10.0/6.022/density)**(1.0/3.0) ;
 cellpar = V**(1.0/3.0)
 #print cellpar

 ##########################################
 # Define distance calculation function
 ##########################################

 ##########################################
 # Random spray
 ##########################################


 tot_attempt = 0;
 for newatom in range(totatom):
    newatom_type = sum([newatom >= sum(natom[0:j+1]) for j in range(nelement)]);
    newatom_position = [ cellpar * random.random() for i in range(3)] ;
    exatom = -1;
    while exatom < newatom:
        tot_attempt = tot_attempt + 1
        if tot_attempt > 100000:     # Exit Loop if it takes too long
            sys.exit()
        newatom_position = [ cellpar * random.random() for i in range(3)] ;
        for exatom in range(newatom+1):
           exatom_type = sum([exatom >= sum(natom[0:j+1]) for j in range(nelement)]);
           dist = dist_pbc(newatom_position,exatom_position[exatom],cellpar) ;
           if dist < cutoff[newatom_type][exatom_type]:
               break
    exatom_position[newatom] = newatom_position ;
 #    print newatom , 'th atom'
        

 ##############################################
 #Writing POSCAR (Converting Fractional to Direct coordinate
 ##############################################


 POSCAR.write(''.join(atomname) + '   density: '+str(density)+'   ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
 POSCAR.write("\n")
 POSCAR.write("   1.0 \n")
 POSCAR.write("{:21.13f}{:19.13f}{:19.13f}\n".format(cellpar,0,0))
 POSCAR.write("{:21.13f}{:19.13f}{:19.13f}\n".format(0,cellpar,0))
 POSCAR.write("{:21.13f}{:19.13f}{:19.13f}\n".format(0,0,cellpar))
 POSCAR.write('   ' + '   '.join(elements))
 POSCAR.write("\n")
 POSCAR.write('    ' + '    '.join([str(x) for x in natom]) + '  \n')
 POSCAR.write('Selective dynamics \n')
 POSCAR.write('Direct \n')

 exatom_position = [[exatom_position[i][j]/cellpar for j in range(3)] for i in range(totatom)]

 for i in range(totatom):
    POSCAR.write("{:19.15f}{:19.15f}{:19.15f}   T   T   T  \n".format(exatom_position[i][0],exatom_position[i][1],exatom_position[i][2]))
# POSCAR.write("{:19.15f}{:19.15f}{:19.15f}   F   F   F  \n".format(exatom_position[totatom-1][0],exatom_position[totatom-1][1],exatom_position[totatom-1][2]))

 for atomi in atomname:
    subprocess.call('cat '+potdir+atomi+'/POTCAR >> POTCAR',shell=True)


def make_oxygen():
  pos = read_poscar_dict("POSCAR_cation")
 
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
        if ii > 22:
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

#t2  = time()
#print(make_oxygen())
#t3 = time()

#print (t3-t2)


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

 printBuffer += " Ta P O\n 16 8 64" 
 printBuffer += '\nCartesian\n'
 for rowCnt in range(numIon):
    for colCnt in range(3):
        printBuffer += '   %12.8f ' %float(Cartesian[rowCnt][colCnt])
    printBuffer += '\n'

 with open(output,"w") as w:
   w.write(printBuffer)

def randspg(composition,elements,volume,tolerance):

  while 1:

    lmin = volume**(1.0/3.0) * 0.5
    lmax = volume**(1.0/3.0) * 1.5

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
