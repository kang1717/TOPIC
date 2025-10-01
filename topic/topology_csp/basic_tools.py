import copy
import subprocess
import os
import shutil
import numpy as np
import math
import yaml
import pymatgen
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.core.structure import IStructure
from topic.topology_csp.basic_tools import *
# basic tools

def atomic_number_to_element(atomic_number):
    elements = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
        21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
        31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
        41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
        51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
        61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
        71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
        81: "Tl", 82: "Pb", 83: "Bi", 84: "Th", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
        91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
        101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
        111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
    }
    return elements.get(atomic_number, "Unknown")

def outcar2positions(f_outcar,step=-1):
    with open(f_outcar,'r') as f:
        outcar = f.readlines()

    # Read atoms
    fake_lib = []
    see__ = 0
    for o in outcar:
        if "POTCAR:    PAW_PBE" in o:
            fake_lib.append(o.split()[2])
            see__ = 1

        if see__ == 1 and '-------------' in o:
            break

    atom_lib = fake_lib[0:int(len(fake_lib)/2)]
 
    # Main loop
    new_iteration    = 0 
    turn_on_position = 0 
    turn_on___       = 0
    volume_turn_on   = 0
    iteration  = 1
    numlist = []

    LATT_ARRAY = []
    COOR_ARRAY = []

    Iteration = 0

    for o in outcar:
        if 'ions per type' in o:
            data = o.split()
            for k in range(4,4+len(atom_lib)):
                numlist.append(data[k])

        if 'Iteration' in o and '(   1)' in o:
            new_iteration = 1

        if 'volume of cell' in o:
            volume_turn_on = 1
            latt = []
        elif volume_turn_on == 1:
            if len(o.split())> 0 and 'direct' not in o and 'length' not in o:
                latt.append([float(o.split()[0]),float(o.split()[1]),float(o.split()[2])])
            elif 'length' in o:
                volume_turn_on = 0
                latt = np.array(latt)



        if new_iteration == 1:
            if 'POSITION' in o:
                turn_on_position = 1

        if turn_on_position == 1 and turn_on___==0:
            if '-------' in o:
                turn_on___ = 1
                iteration += 1
                coor = []
        elif turn_on___ == 1:
            if '--' not in o:
                coor.append([float(o.split()[0]),float(o.split()[1]),float(o.split()[2])])
            else:
                turn_on___ = 0
                turn_on_position = 0
                new_iteration = 0
                coor = np.array(coor)
                COOR_ARRAY.append(coor)
                LATT_ARRAY.append(latt)

    # change into the array form
    for i in range(len(numlist)):
        numlist[i] = int(numlist[i])
    numlist = np.array(numlist)
    atomlist = np.array(atom_lib)
    atomarray,numarray = make_atom_array(atomlist,numlist)

    pos_dicts = []
    for i in range(len(COOR_ARRAY)):
        pos_dict = {}
        pos_dict['latt'] = LATT_ARRAY[i]
        pos_dict['coor'] = COOR_ARRAY[i]
        pos_dict['atomlist'] = atomlist
        pos_dict['numlist'] = numlist
        pos_dict['fix'] = np.array([['T','T','T'] for i in range(len(COOR_ARRAY[i]))])
        pos_dict['cartesian'] = True
        pos_dict['selective_dynamics'] = True
        pos_dict['atomarray'] = atomarray
        pos_dict['numarray'] = numarray
        pos_dicts.append(pos_dict)

    return pos_dicts

def distance_fast(pos,i,j):
    coor_ = shift2center(pos['coor'],pos['latt'],i)
    coor__= get_into_the_lattice(coor_,pos['latt'])

    return np.linalg.norm(coor__[i]-coor__[j])

def shift2center(coor,latt,target):
    center = latt[0]/2+latt[1]/2+latt[2]/2
    coor_ = copy.deepcopy(coor)
    for i in range(len(coor_)):
        coor_[i] -= coor[target] + center
    return coor_

def get_into_the_lattice(coor,latt):

  for i in range(len(coor)):

    for j in range(3):

      while coor[i][j] >= latt[j][j] or coor[i][j] < 0.0:
      
        if coor[i][j] < 0:
          coor[i][j] += latt[j][j]

        if coor[i][j] >= latt[j][j]:
          coor[i][j] -= latt[j][j]

  return coor


def getE(outcar_file):
  with open(outcar_file,"r") as f:
   outcar = f.readlines()
  l = len(outcar)
  numarray = [l-1-i for i in range(l)]
  E = 1000000
  for i in numarray:
    if 'free  ' in outcar[i]:
     E = float(outcar[i].split()[4])
     break
  return E

def getE_first(outcar_file):
  with open(outcar_file,"r") as f:
   outcar = f.readlines()
  l = len(outcar)
  numarray = [l-1-i for i in range(l)]
  E = 1000000
  for i in range(len(outcar)):
    if 'free  ' in outcar[i]:
     E = float(outcar[i].split()[4])
     break
  return E

def magmom_from_previous(outcar_file):
  with open(outcar_file,"r") as f:
    outcar = f.readlines()

  line1 = 0
  line2 = 0
  for i in range(len(outcar)):
    if "magnetization (x)" in outcar[i]:
      line1 = i
    if "-----------" in outcar[i]:
      line2 = i
    if "NIONS" in outcar[i]:
      nion = int(outcar[i].split()[11])

  line2 = line1 + nion+4


  magmom = []

  for i in range(line1+4,line2):
    magmom.append(float(outcar[i].split()[-1]))

  with open("INCAR","r") as f:
    incar = f.readlines()

  with open("INCAR","w") as f:
    for incarline in incar:
      if "MAGMOM" not in incarline:
        f.write(incarline)
    
    f.write("MAGMOM = ")
    for i in range(len(magmom)):
      f.write("  {}  ".format(magmom[i]))
    f.write("\n")


def read_poscar(filename):

  with open(filename,"r") as f:
    poscar = f.readlines()

  if 1:
    latt = []
    for i in range(2,5):
      latt.append(list(map(float,poscar[i].split())))
    latt = np.array(latt)

    atomlist = []
    atomlist = poscar[5].split()

    numlist  = []
    numlist  = list(map(int,poscar[6].split()))

    N = sum(numlist)

    if 'Selective' in poscar[7]:
        selective_dynamics = True
    else:
        selective_dynamics = False

    if selective_dynamics == True:
        cartesian_line = 8
    else:
        cartesian_line = 7

    if "Cartesian" in poscar[cartesian_line]:
        cartesian = True
    else:
        cartesian = False

    coor = []
    fix  = []
    for i in range(cartesian_line+1,N+cartesian_line+1):
      p = poscar[i]
      coor.append(list(map(float,p.split()[0:3])))
      if len(p.split())>4:
        fix.append(p.split()[3:6])
    coor = np.array(coor)
 
    atomlist = np.array(atomlist)
    numlist  = np.array(numlist)

    return latt,coor,atomlist,numlist, fix,cartesian,selective_dynamics 

def read_poscar_dict(filename):

  with open(filename,"r") as f:
    poscar = f.readlines()

  if 1:
    latt = []
    for i in range(2,5):
      latt.append(list(map(float,poscar[i].split())))
    latt = np.array(latt)

    atomlist = []
    atomlist = poscar[5].split()

    numlist  = []
    numlist  = list(map(int,poscar[6].split()))

    N = sum(numlist)

    if 'Selective' in poscar[7]:
        selective_dynamics = True
    else:
        selective_dynamics = False

    if selective_dynamics == True:
        cartesian_line = 8
    else:
        cartesian_line = 7

    if "Cartesian" in poscar[cartesian_line]:
        cartesian = True
    else:
        cartesian = False

    coor = []
    fix  = []
    for i in range(cartesian_line+1,N+cartesian_line+1):
      p = poscar[i]
      coor.append(list(map(float,p.split()[0:3])))
      if len(p.split())>4:
        fix.append(p.split()[3:6])
    coor = np.array(coor)
 
    atomlist = np.array(atomlist)
    numlist  = np.array(numlist)

    return_dict = {}
    return_dict['latt'] = latt
    return_dict['coor'] = coor
    return_dict['atomlist'] = atomlist
    return_dict['numlist'] = numlist
    return_dict['fix'] = fix
    return_dict['cartesian'] = cartesian
    return_dict['selective_dynamics'] = selective_dynamics
    atomarray,numarray = make_atom_array(atomlist,numlist)
    return_dict['atomarray'] = atomarray
    return_dict['numarray']  = numarray
    return return_dict 

def empty_poscar_dict(latt):
    return_dict = {}
    return_dict['latt'] = latt
    return_dict['coor'] = np.array([[]])
    return_dict['atomlist'] = np.array([])
    return_dict['numlist'] = np.array([],dtype=int)
    return_dict['fix'] = np.array([])
    return_dict['cartesian'] = True
    return_dict['selective_dynamics'] = False
    return_dict['atomarray'] = np.array([])
    return_dict['numarray']  = np.array([])
    return return_dict

#put atoms in the box
def put_molecule(poscar,poscar_molecule):
    
    new_poscar = poscar.copy()
    
    coor_m = poscar_molecule['coor']
    atom_m = poscar_molecule['atomarray']

    test_i = 0
    for c,a in zip(coor_m,atom_m):
        test_i += 1

        
        new_poscar = adatom_dict_fix(new_poscar,c,a)
    return new_poscar


def adatom_dict_fix(poscar,target_coor,element,fix='T'):

  new_poscar = poscar.copy()

  atomlist = poscar['atomlist']
    
  if element not in poscar['atomlist']:
    new_poscar['atomlist'] = np.append(poscar['atomlist'],element)
    new_poscar['numlist']  = np.append(poscar['numlist'],1)
    n1,n2 =  np.shape(new_poscar['coor'])
    n = n1*n2
    
    if n == 0:
      new_poscar['coor']     = np.array([target_coor])
      new_poscar['fix']      = np.array([np.array([fix,fix,fix])])
    else:
      new_poscar['coor']     = np.vstack((poscar['coor'],target_coor))
      new_poscar['fix']      = np.vstack((poscar['fix'],np.array([fix,fix,fix])))

  else:
    N = 0
    for i,a in enumerate(atomlist):
      N += poscar['numlist'][i]
      if a == element:
        element_num = i
        break
    
    N = int(N)
    new_poscar['coor']     = np.insert(poscar['coor'],N,target_coor,axis=0)
    new_poscar['fix']      = np.insert(poscar['fix'], N, np.array([fix,fix,fix]), axis=0)

    new_poscar['numlist'][element_num] += 1


    atomarray,numarray = make_atom_array(new_poscar['atomlist'],new_poscar['numlist'])
    new_poscar['atomarray'] = atomarray
    new_poscar['numarray'] = numarray
  return new_poscar

def adatom_dict_fix_new(poscar,target_coor,element,fix='T'):

  new_poscar = poscar.copy()

  atomlist = poscar['atomlist']
    
  if element not in poscar['atomlist']:
    new_poscar['atomlist'] = np.append(poscar['atomlist'],element)
    new_poscar['numlist']  = np.append(poscar['numlist'],1)
    n1,n2 =  np.shape(new_poscar['coor'])
    n = n1*n2
    
    if n == 0:
      new_poscar['coor']     = np.array([target_coor])
      new_poscar['fix']      = np.array([np.array([fix,fix,fix])])
    else:
      new_poscar['coor']     = np.vstack((poscar['coor'],target_coor))
      new_poscar['fix']      = np.vstack((poscar['fix'],np.array([fix,fix,fix])))

  else:
    N = 0
    for i,a in enumerate(atomlist):
      N += poscar['numlist'][i]
      if a == element:
        element_num = i
        break
    
    N = int(N)
    if element_num == (len(poscar['numlist'])-1):
      new_poscar['coor']     = np.insert(poscar['coor'], N-1, target_coor,axis=0) 
      new_poscar['fix']      = np.insert(poscar['fix'], N-1, np.array([fix,fix,fix]), axis=0)
    else:
      new_poscar['coor']     = np.insert(poscar['coor'],N,target_coor,axis=0)
      new_poscar['fix']      = np.insert(poscar['fix'], N, np.array([fix,fix,fix]), axis=0)

    new_poscar['numlist'][element_num] += 1


    atomarray,numarray = make_atom_array(new_poscar['atomlist'],new_poscar['numlist'])
    new_poscar['atomarray'] = atomarray
    new_poscar['numarray'] = numarray
  return new_poscar

def make_supercell(latt,coor,atomlist,numlist,nx,ny,nz,cartesian=True):

  Natom = len(coor)

  n = np.array([float(nx),float(ny),float(nz)])
  N = nx*ny*nz

  if cartesian == False:
    coor = direct2cartesian(latt,coor)

  # change lattice
  latt_new = np.array([[0.0,0.0,0.0] for i in range(3)])
  for i in range(3):
    for j in range(3):
      latt_new[i][j] = latt[i][j] * n[i]

  # change numlist
  numlist_new = [int(atomnum)*int(N) for atomnum in numlist]

  # change coordinate
  coor_new = np.array([[0.0,0.0,0.0] for i in range(N*Natom)])
  index = -1
  for a in range(Natom):
    for i in range(nx):
      for j in range(ny):
        for k in range(nz):
          index += 1
          coor_new[index][0] = coor[a][0] + float(i)*latt[0][0] + float(j)*latt[1][0] + float(k)*latt[2][0]
          coor_new[index][1] = coor[a][1] + float(i)*latt[0][1] + float(j)*latt[1][1] + float(k)*latt[2][1]
          coor_new[index][2] = coor[a][2] + float(i)*latt[0][2] + float(j)*latt[1][2] + float(k)*latt[2][2]

  latt_new = np.array(latt_new)
  coor_new = np.array(coor_new)

  return latt_new, coor_new, np.array(atomlist), np.array(numlist_new)

def make_supercell_dict(dat,nx,ny,nz):

  latt = dat['latt']
  coor = dat['coor']
  atomlist = dat['atomlist']
  numlist  = dat['numlist']
  cartesian= dat['cartesian']
  fix = dat['fix']
  Natom = len(coor)

  n = np.array([float(nx),float(ny),float(nz)])
  N = nx*ny*nz

  if cartesian == False:
    coor = direct2cartesian(latt,coor)

  # change lattice
  latt_new = np.array([[0.0,0.0,0.0] for i in range(3)])
  for i in range(3):
    for j in range(3):
      latt_new[i][j] = latt[i][j] * n[i]

  # change numlist
  numlist_new = [int(atomnum)*int(N) for atomnum in numlist]

  # change coordinate
  coor_new = np.array([[0.0,0.0,0.0] for i in range(N*Natom)])
  index = -1
  fix_new  = []
  for a in range(Natom):
    for i in range(nx):
      for j in range(ny):
        for k in range(nz):
          index += 1
          coor_new[index][0] = coor[a][0] + float(i)*latt[0][0] + float(j)*latt[1][0] + float(k)*latt[2][0]
          coor_new[index][1] = coor[a][1] + float(i)*latt[0][1] + float(j)*latt[1][1] + float(k)*latt[2][1]
          coor_new[index][2] = coor[a][2] + float(i)*latt[0][2] + float(j)*latt[1][2] + float(k)*latt[2][2]
          fix_new.append(fix[a])
  latt_new = np.array(latt_new)
  coor_new = np.array(coor_new)

  dat_new = dat.copy()
  dat_new['latt'] = latt_new
  dat_new['coor'] = coor_new
  dat_new['atomlist'] = np.array(atomlist)
  dat_new['numlist']  = np.array(numlist_new)
  dat_new['fix'] = np.array(fix_new)
  dat_new['cartesian'] = True
  return dat_new 

def add_magmom(incar_name,magarr,Uarr):
  with open(incar_name,"r") as f:
    incar = f.readlines()

  with open(incar_name,"w") as f:
    for fstr in incar:
      if "MAGMOM" in fstr:
        f.write(fstr.split("\n")[0])
        for add in magarr:
          f.write(" {} ".format(add))
        f.write("\n")

      elif "LDAUL" in fstr:
        f.write(fstr.split("\n")[0])
        for add in Uarr:
          if float(add) == 0.0:
            f.write(" -1 ")
          else:
            f.write(" 2 ")
        f.write("\n")

      elif "LDAUU" in fstr:
        f.write(fstr.split("\n")[0])
        for add in Uarr:
          f.write(" {} ".format(add))
        f.write("\n")

      elif "LDAUJ" in fstr:
        f.write(fstr.split("\n")[0])
        for add in Uarr:
          f.write(" 0.0 ")
        f.write("\n")
 
      else:
        f.write(fstr)

def calculate_distance(coor_i,coor_j,latt):

  distance = 10000.0

  for I in range(-1,2):
   for J in range(-1,2):
    for K in range(-1,2):
      i=  float(I); j = float(J); k = float(K)

      coor_i_ = coor_i + latt[0]*i + latt[1]*j + latt[2]*k

      temp = np.linalg.norm(coor_i_-coor_j)
      if temp < distance:
        distance = temp

  return distance
      

def direct2cartesian(latt,coor):
  return np.dot(coor,latt)

def cartesian2direct(latt,coor):
  return np.dot(coor,np.linalg.inv(latt))

def distinct_magnetic_moments():
    
    # file read
    
    with open("OUTCAR","r") as fo:
        outcar  = fo.readlines()
    
    with open("CONTCAR","r") as fc:
        contcar = fc.readlines()

    ################# MAGNETIZATION #################
    # read magnetization from OUTCAR
    magline = 0
    for i in range(len(outcar)):
        if "magnetization (x)" in outcar[i]:
            magline = i            
    
    if magline != 0:
        start = 0
        for i in range(magline,len(outcar)):
            if "------" in outcar[i] and start == 0:
                mag = []
                start = 1
            
            elif start == 1 and "-------" not in outcar[i]:
                mag.append(float(outcar[i].split()[-1]))
            
            elif start == 1 and "--------" in outcar[i]:
                break
            
        # read atomic information from CONTCAR
        atomic_species = contcar[5].split()
        atomic_numbers = list(map(int,contcar[6].split()))
    
        atom_array = []
        for i in range(len(atomic_species)):
            atom_array += atomic_numbers[i]*[atomic_species[i]]
        
        for i in range(len(atom_array)):
            if mag[i] < 0 and abs(mag[i]) > 0.5 and atom_array[i] in mag_atom_list:
                atom_array[i] = fake_atom_lib(atom_array[i])
            
        # print new but unordered CONTCAR
        with open("CONTCAR_unordered","w") as fw:
            for i in range(len(contcar)):
                if i == 5:
                    for atom in atom_array:
                        fw.write(atom)
                        fw.write(" ")
                    fw.write("\n")
                
                elif i == 6:
                    for atom in atom_array:
                        fw.write(" 1 ")
                    fw.write("\n")
            
                else:
                    fw.write(contcar[i])
                
        # get ordered structure using pymatgen
        unordered_contcar = Structure.from_file("CONTCAR_unordered").get_sorted_structure()
        IStructure.to(unordered_contcar,"poscar",filename="CONTCAR_mag")
    
        # write magnetic moment log
        mag_dict = {}
        num_dict = {}
    
        for i in range(len(atom_array)):
            if atom_array[i] not in mag_dict.keys():
                mag_dict[atom_array[i]] = mag[i]
                num_dict[atom_array[i]] = 1
            
            else:
                mag_dict[atom_array[i]] += mag[i]
                num_dict[atom_array[i]] += 1
        
        
        with open("magmom_log","w") as fm:
            for atom in mag_dict.keys():
                magmom = mag_dict[atom]/num_dict[atom]
                if abs(magmom) < 0.5:  
                    magmom = 0
                fm.write("%s: %f\n"%(atom,magmom))

    else:
        with open("magmom_log","w") as f:
            for atom in atomic_species:
                magmom = 0
                fm.write("%s: %f\n"%(atom,magmom))
                   
                    
    ################# U correction #################
    # read LDAU
    ldau_yes = 0
    ldau_L    = []
    ldau_U    = []
    ldau_J    = []
    for i in range(len(outcar)):
        if "LDAUTYPE =  2" in outcar[i]:
            ldau_yes = 1
            
        elif "LDAUL" in outcar[i]:
            ldau_L = list(map(int,outcar[i].split()[7:]))
            
        elif "LDAUU" in outcar[i]:
            ldau_U = list(map(float,outcar[i].split()[7:]))
            
        elif "LDAUJ" in outcar[i]:
            ldau_J = list(map(float,outcar[i].split()[7:]))
            
        if ldau_L != [] and ldau_U != [] and ldau_J != []:
            break

    if ldau_yes == 1:
        Ueff = [ldau_U[i] - ldau_J[i] for i in range(len(ldau_U))]
    else:
        Ueff = [0 for i in range(len(atomic_species))]
        
    # write LDAU
    with open("ldau_log","w") as fl:
        if ldau_yes == 1:
            for i in range(len(atomic_species)):
                fl.write("%s: %f\n"%(atomic_species[i],Ueff[i]))
        else:
            for i in range(len(atomic_species)):
                fl.write("%s: 0\n"%(atomic_species[i]))

def add_array(arr, add, i): 

  if i == -1: 
    return np.append([add],arr,axis=0)

  elif i == len(arr)-1:
    return np.append(arr,[add],axis=0)

  else:
    tmp = np.append(arr[0:i+1],[add],axis=0)
    return np.append(tmp,arr[i+1:len(arr)],axis=0)

def remove_array(arr,i):

  l = len(arr)

  if i == 0:
    return arr[1:l]
  elif i == l-1:
    return arr[0:l-1]
  else:
    return np.append(arr[0:i],arr[i+1:l],axis=0)

                
def make_atom_array(atomlist,numlist):
  N = sum(numlist)

  atomarray = []
  for i in range(len(numlist)):
    for j in range(numlist[i]):
      atomarray.append(atomlist[i])

  numarray = []
  for i in range(len(numlist)):
    for j in range(numlist[i]):
      numarray.append(i)

  return np.array(atomarray), np.array(numarray)

def remove_atom(latt,coor,atomlist,numlist,element):
    
    atomarray, numarray = make_atom_array(atomlist,numlist)
    
    numlist0 = np.copy(numlist)
    
    for i in range(len(coor)):
        if atomarray[i] == element:
            
            coor_r = np.delete(coor, i, axis = 0)
            
            for j in range(len(atomlist)):
                if atomlist[j] == element:
                    numlist0[j] -= 1
            break
            
    return latt, np.copy(coor_r), np.copy(atomlist), numlist0

def adatom_dict(poscar,target_coor,element,fix='T'):

  new_poscar = poscar.copy()

  atomlist = poscar['atomlist']
    
  if element not in poscar['atomlist']:
    new_poscar['atomlist'] = np.append(poscar['atomlist'],element)
    new_poscar['numlist']  = np.append(poscar['numlist'],1)
    new_poscar['coor']     = np.append(poscar['coor'],target_coor)
    if new_poscar['selective_dynamics'] == True:
      new_poscar['fix']      = np.append(poscar['fix'],np.array([fix,fix,fix]))

  else:
    N = 0
    for i,a in enumerate(atomlist):
      N += poscar['numlist'][i]
      if a == element:
        element_num = i
        break

    new_poscar['coor']     = np.insert(poscar['coor'], N, target_coor,axis=0) 
    new_poscar['numlist'][element_num] += 1
    
    if new_poscar['selective_dynamics'] == True:
      new_poscar['fix']      = np.insert(poscar['fix'], N, np.array([fix,fix,fix]), axis=0)

  return new_poscar

def delete_distant_atoms(poscar,cutoff=0.1):

  erase = []
  coor   = copy.deepcopy(poscar['coor'])
  latt   = copy.deepcopy(poscar['latt'])
  for i in range(len(coor)):
    distant = 0 
    for j in range(i+1,len(coor)):
      distance = calculate_distance(coor[i],coor[j],latt)
      if distance < cutoff:
        distant = 1 


    if distant == 1:
      erase.append(i)

  save = 0 
  for e in erase:
    poscar = remove_atom_dict(poscar,e-save)
    save += 1

  return poscar


def remove_atom_dict(dat,number):

    latt = dat['latt']
    coor = dat['coor']
    atomlist = dat['atomlist']
    numlist  = dat['numlist']

    atomarray, numarray = make_atom_array(atomlist,numlist)
    
    numlist0 = np.copy(numlist)
    
    for i in range(len(coor)):
        if i == number:
            
            coor_r = np.delete(coor, i, axis = 0)
            
            for j in range(len(atomlist)):
                if atomlist[j] == atomarray[number]:
                    numlist0[j] -= 1
            break
           
    dat_new = dat.copy()
    dat_new['coor'] = coor_r
    dat_new['atomlist'] = atomlist
    dat_new['numlist']  = numlist0
    dat_new['atomarray'] = [atom for atom, number in zip(dat_new['atomlist'], dat_new['numlist']) for _ in range(number)]
    dat_new['numarray'] = [atom for atom, number in zip(range(len(dat_new['numlist'])), dat_new['numlist']) for _ in range(number)]
 
    return dat_new 

def write_contcar(latt,coor,atomlist,numlist,filename="CONTCAR_out",cartesian=True):
  with open(filename,"w") as f:
    f.write("make_exsolution.py\n")
    f.write("1.0000000000000000\n")

    for i in range(3):
      for j in range(3):
        f.write(" %.15f "%(latt[i][j]))
      f.write("\n")

    for atom in atomlist:
      f.write(" %s "%(atom))
    f.write("\n")

    for atomnum in numlist:
      f.write(" %d "%(atomnum))
    f.write("\n")

    if cartesian==True:
      f.write("Cartesian\n")
    else:
      f.write("Direct\n")
    for coor0 in coor:
      #print (coor0)
      f.write(" %.15f %.15f %.15f\n"%(coor0[0],coor0[1],coor0[2]))

def write_contcar_fix(latt,coor,atomlist,numlist,fix,filename="CONTCAR_out",cartesian=True):
  with open(filename,"w") as f:
    f.write("make_exsolution.py\n")
    f.write("1.0000000000000000\n")

    for i in range(3):
      for j in range(3):
        f.write(" %.15f "%(latt[i][j]))
      f.write("\n")

    for atom in atomlist:
      f.write(" %s "%(atom))
    f.write("\n")

    for atomnum in numlist:
      f.write(" %d "%(atomnum))
    f.write("\n")

    f.write("Selective dynamics\n")
    if cartesian==True:
      f.write("Cartesian\n")
    else:
      f.write("Direct\n")

    #print (coor)
    #print (fix)

    i = 0
    for coor0 in coor:
      f.write(" %.15f %.15f %.15f %s %s %s\n"%(coor0[0],coor0[1],coor0[2],fix[i][0],fix[i][1],fix[i][2]))
      i+=1

def write_contcar_dict(return_dict,filename="CONTCAR_out"):
  latt = return_dict['latt']
  coor = return_dict['coor'] 
  atomlist = return_dict['atomlist'] 
  numlist = return_dict['numlist'] 
  fix = return_dict['fix'] 
  cartesian = return_dict['cartesian'] 

  with open(filename,"w") as f:
    f.write("make_exsolution.py\n")
    f.write("1.0000000000000000\n")

    for i in range(3):
      for j in range(3):
        f.write(" %.15f "%(latt[i][j]))
      f.write("\n")

    for atom in atomlist:
      f.write(" %s "%(atom))
    f.write("\n")

    for atomnum in numlist:
      f.write(" %d "%(atomnum))
    f.write("\n")
    if len(fix) > 0:
      f.write("Selective dynamics\n")
    else:
      asdf = 1
    if cartesian==True:
      f.write("Cartesian\n")
    else:
      f.write("Direct\n")

    #print (coor)
    #print (fix)

    i = 0
    for coor0 in coor:
      if len(fix) > 0:
        f.write(" %.15f %.15f %.15f %s %s %s\n"%(coor0[0],coor0[1],coor0[2],fix[i][0],fix[i][1],fix[i][2]))
        i+=1
      else:
        f.write(" %.15f %.15f %.15f\n"%(coor0[0],coor0[1],coor0[2]))
        i+=1
 
def make_kpoints_vacancy(folder,homepath,scaling):
    
    with open(homepath+"/Done/"+folder+"/relax_GGA/KPOINTS","r") as f:
        
        kpt_float = np.array(list(map(float,f.readlines()[3].split())))
        
    kpt_float = kpt_float / scaling
    
    kpt = np.array([math.ceil(i) for i in kpt_float])
    
    with open("KPOINTS","w") as f:
        
        f.write("Auto k-point\n")
        f.write("0\n")
        f.write("Monk-horst\n")
        f.write(" {} {} {}\n".format(kpt[0],kpt[1],kpt[2]))
        f.write(" 0 0 0")
        

def read_kpoints(filename ="KPOINTS"):
  with open(filename,"r") as f:
    dat = f.readlines()[3].split()

  dat[0] = int(dat[0])
  dat[1] = int(dat[1])
  dat[2] = int(dat[2])

  return dat

def write_kpoints(kpt,filename="KPOINTS"):
    with open(filename,"w") as f:
        
        f.write("Auto k-point\n")
        f.write("0\n")
        f.write("Monk-horst\n")
        f.write(" {} {} {}\n".format(kpt[0],kpt[1],kpt[2]))
        f.write(" 0 0 0")

specific_potcar = {}
specific_potcar['Ca'] = 'Ca_sv'
specific_potcar['Sr'] = 'Sr_sv'
specific_potcar['Ba'] = 'Ba_sv'
specific_potcar['Zr'] = 'Zr_sv'
specific_potcar['Ni'] = 'Ni_pv'
specific_potcar['Cr'] = 'Cr_pv'
specific_potcar['Cu'] = 'Cu_pv'
specific_potcar['Fe'] = 'Fe_pv'
specific_potcar['Mg'] = 'Mg_pv'
specific_potcar['Mn'] = 'Mn_pv'
specific_potcar['Rh'] = 'Rh_pv'
specific_potcar['Ti'] = 'Ti_pv'
specific_potcar['Ru'] = 'Ru_pv'
specific_potcar['Mo'] = 'Mo_pv'


def write_potcar(poscar,potcar_dir):

    with open(poscar,"r") as f:
        for i in range(5):
            f.readline()

        atomname = f.readline().split()

    start = -1
    for atom in atomname:

        if atom in fake_atom_list:
            atom = real_atom_lib(atom)
        
        start += 1
        if atom not in specific_potcar.keys():
            potname = atom
        else:
            potname = specific_potcar[atom]

        if start == 0:
            subprocess.call('cat %s/%s/POTCAR >  POTCAR'%(potcar_dir,potname),shell=True)
        else:
            subprocess.call('cat %s/%s/POTCAR >> POTCAR'%(potcar_dir,potname),shell=True)
            
def write_magmom(bulk_path,poscar,fw):

    # read bulk magmom
    with open(bulk_path+"/magmom_log","r") as f:
        mag_info = yaml.safe_load(f)

    # read POSCAR
    with open(poscar,"r") as f:
        fposcar = f.readlines()
        atomic_species = fposcar[5].split()
        atomic_numbers = list(map(int,fposcar[6].split()))

    # write MAGMOM
    fw.write("\n")
    fw.write("   MAGMOM = ")

    i = -1
    for atom in atomic_species:
        i += 1
        if atom in mag_info.keys():
            fw.write(" %d*%f "%(atomic_numbers[i],mag_info[atom]))
        else:
            fw.write(" %d*0 "%(atomic_numbers[i]))
    fw.write("\n")

def write_U(bulk_path, poscar, fw):

    # read bulk u parameters
    with open(bulk_path+"/ldau_log","r") as f:
        ldau_info = yaml.safe_load(f)

    # write ldau for fake atoms
    fake_ldau = {}
    for atom in ldau_info.keys():
        if atom in mag_atom_list:
            fake_ldau[fake_atom_lib(atom)] = ldau_info[atom]

    for atom in fake_ldau:
        ldau_info[atom] = fake_ldau[atom]

    # read POSCAR
    with open(poscar,"r") as f:
        fposcar = f.readlines()
        atomic_species = fposcar[5].split()
        atomic_numbers = list(map(int,fposcar[6].split()))

    # determine if +U is used
    ldau_used = 0
    for atom in ldau_info.keys():
        if ldau_info[atom] > 0:
            ldau_used = 1


    # write LDAU
    if ldau_used == 1:

        fw.write("\n")
        fw.write("   LDAU = .TRUE.\n")
        fw.write("   LDAUTYPE = 2\n")

        fw.write("   LDAUL = ")
        i = -1
        for atom in atomic_species:
            i += 1
            if atom in ldau_info.keys():
                if ldau_info[atom] > 0:
                    fw.write(" 2 ")
                else:
                    fw.write(" -1 ")
            else:
                fw.write(" -1 ")
        fw.write("\n")

        fw.write("   LDAUU = ")
        i = -1
        for atom in atomic_species:
            i += 1
            if atom in ldau_info.keys():
                if ldau_info[atom] > 0:
                    fw.write(" %f "%(ldau_info[atom]+1.0))
                else:
                    fw.write(" 0.0 ")
            else:
                fw.write(" 0.0 ")
        fw.write("\n")

        fw.write("   LDAUJ = ")
        i = -1
        for atom in atomic_species:
            i += 1
            if atom in ldau_info.keys():
                if ldau_info[atom] > 0:
                    fw.write(" 1.0 ")
                else:
                    fw.write(" 0.0 ")
            else:
                fw.write(" 0.0 ")
        fw.write("\n")

        fw.write("   LDAUPRINT = 0\n")
        fw.write("   LMAXMIX = 4\n")
        
def write_nbands(fw,numlist,multiply=2.0):

    potcar = str(subprocess.check_output("grep ZVAL POTCAR",shell=True))
    potcar = potcar.split()

    zval = []
    for i in range(len(potcar)-2):
        if potcar[i] == 'ZVAL':
            zval.append(int((float(potcar[i+2]))))

    NELECT = 0 
    for i in range(len(zval)):
        NELECT += zval[i]*numlist[i]

    NIONS = sum(numlist)

    NBANDS = max(int(NELECT/2+NIONS/2),int(NELECT*0.6))

    fw.write("  NBANDS = %d\n"%(int(NBANDS*multiply)))

def write_incar(folder,homepath,npar,kpar,numlist,poscar):
    
    fw = open("INCAR","w")
    
    with open(homepath+"/Done/"+folder+"/relax_GGA/INCAR",'r') as f:
        
        bulk_incar_lines = f.readlines()
        
    # write incar
    for fstr in bulk_incar_lines:
        if "LREAL" in fstr:
            continue
        elif "PREC" in fstr:
            continue
        elif "NSW" in fstr:
            continue
        elif "ISIF" in fstr:
            continue
        elif "SIGMA" in fstr:
            continue
        elif "NELMIN" in fstr:
            continue
        elif "NELMDL" in fstr:
            continue
        elif " NELM " in fstr:
            continue
        elif "EDIFF" in fstr:
            continue
        elif "ISPIN" in fstr:
            continue
        elif "ISYM" in fstr:
            continue
        elif "NPAR" in fstr:
            continue
        elif "KPAR" in fstr:
            continue
        elif "MAGMOM" in fstr:
            continue
        elif "LDAU" in fstr:
            continue
        elif "LDAUTYPE" in fstr:
            continue
        elif "LDAUL" in fstr:
            continue
        elif "LDAUU" in fstr:
            continue
        elif "LDAUJ" in fstr:
            continue
        elif "LDAUPRINT" in fstr:
            continue
        elif "LMAXMIX" in fstr:
            continue
        elif "POTIM" in fstr:
            continue
        else:
            fw.write(fstr)

    fw.write("\n")
    fw.write(" defaults: \n")  
    fw.write("   PREC  = Normal\n")
    fw.write("   LREAL  = Auto\n")
    fw.write("   NSW    = 200\n")
    fw.write("   ISIF   = 2\n")
    fw.write("   SIGMA  = 0.05\n")
    fw.write("   NELM    = 500\n")
    fw.write("   NELMIN = 4\n")
    fw.write("   NELMDL = -10\n")
    fw.write("   ISPIN  = 2\n")
    fw.write("   ISYM   = 0\n")
    fw.write("   POTIM   = 0.1\n")
    fw.write("   EDIFF  = 1E-05\n")
    fw.write("\n")
    fw.write("NPAR = %d\n"%(npar))
    fw.write("KPAR = %d\n"%(kpar))
    fw.write("\n")

    bulk_path = homepath+"/Done/"+folder+"/relax_GGA/"
    write_magmom(bulk_path, poscar,fw)
    write_U(bulk_path, poscar,fw)
    
    write_nbands(fw,numlist)

def write_incar_isif(folder,homepath,npar,kpar,numlist,poscar):
    
    fw = open("INCAR","w")
    
    with open(homepath+"/Done/"+folder+"/relax_GGA/INCAR",'r') as f:
        
        bulk_incar_lines = f.readlines()
        
    # write incar
    for fstr in bulk_incar_lines:
        if "LREAL" in fstr:
            continue
        elif "PREC" in fstr:
            continue
        elif "NSW" in fstr:
            continue
        elif "ISIF" in fstr:
            continue
        elif "SIGMA" in fstr:
            continue
        elif "NELMIN" in fstr:
            continue
        elif "NELMDL" in fstr:
            continue
        elif "EDIFF" in fstr:
            continue
        elif "ISPIN" in fstr:
            continue
        elif "ISYM" in fstr:
            continue
        elif "NPAR" in fstr:
            continue
        elif "KPAR" in fstr:
            continue
        elif "MAGMOM" in fstr:
            continue
        elif "LDAU" in fstr:
            continue
        elif "LDAUTYPE" in fstr:
            continue
        elif "LDAUL" in fstr:
            continue
        elif "LDAUU" in fstr:
            continue
        elif "LDAUJ" in fstr:
            continue
        elif "LDAUPRINT" in fstr:
            continue
        elif "LMAXMIX" in fstr:
            continue
        elif "POTIM" in fstr:
            continue
        else:
            fw.write(fstr)

    fw.write("\n")
    fw.write(" defaults: \n")  
    fw.write("   PREC  = Normal\n")
    fw.write("   LREAL  = Auto\n")
    fw.write("   NSW    = 200\n")
    fw.write("   ISIF   = 3\n")
    fw.write("   SIGMA  = 0.05\n")
    fw.write("   NELMIN = 4\n")
    fw.write("   NELMDL = -10\n")
    fw.write("   ISPIN  = 2\n")
    fw.write("   ISYM   = 0\n")
    fw.write("   POTIM   = 0.1\n")
    fw.write("   EDIFF  = 1E-05\n")
    fw.write("\n")
    fw.write("NPAR = %d\n"%(npar))
    fw.write("KPAR = %d\n"%(kpar))
    fw.write("\n")

    bulk_path = homepath+"/Done/"+folder+"/relax_GGA/"
    write_magmom(bulk_path, poscar,fw)
    write_U(bulk_path, poscar,fw)
    
    write_nbands(fw,numlist)
# FAKE atoms for magnetic calculations
mag_atom_list = [ 'V','Cr','Mn','Fe','Co','Ni','Cu']
fake_atom_list= ['Np','Pu','Am','Cm','Bk','Cf','Es']

def fake_atom_lib(atom):
    
    fake_lib = {}
    
    fake_lib['V']  = 'Np'
    fake_lib['Cr'] = 'Pu'
    fake_lib['Mn'] = 'Am'
    fake_lib['Fe'] = 'Cm'
    fake_lib['Co'] = 'Bk'
    fake_lib['Ni'] = 'Cf'
    fake_lib['Cu'] = 'Es'
    
    return fake_lib[atom]

def real_atom_lib(atom):
    
    fake_lib = {}
    
    fake_lib['Np'] = 'V'
    fake_lib['Pu'] = 'Cr'
    fake_lib['Am'] = 'Mn'
    fake_lib['Cm'] = 'Fe'
    fake_lib['Bk'] = 'Co'
    fake_lib['Cf'] = 'Ni'
    fake_lib['Es'] = 'Cu'
    
    return fake_lib[atom]

def run_vasp(homepath,folder,job,hour='default'):
        ### write batch
        with open(homepath+"/vasp.j","r") as f:
            batch = f.readlines()
            
        with open("batch.sh","w") as w:
            for fstr in batch:
              if hour == '2hr':
                if "#BSUB -J" in fstr:
                    w.write("#BSUB -J {}{}\n".format(folder,job))
                elif "BSUB -q" in fstr:
                    sentence = fstr.split("_")[0]+"_2hr\n"
                    w.write(sentence)
                else:
                    w.write(fstr)
         
              else:
                if "#BSUB -J" in fstr:
                    w.write("#BSUB -J {}{}\n".format(folder,job))
                else:
                    w.write(fstr)
                    
        subprocess.call("bsub < batch.sh",shell=True)

def run_vasp_re(homepath,folder,job,hour='default'):
        ### write batch
        with open(homepath+"/vasp_re.j","r") as f:
            batch = f.readlines()
            
        with open("batch.sh","w") as w:
            for fstr in batch:
              if hour == '2hr':
                if "#BSUB -J" in fstr:
                    w.write("#BSUB -J {}{}\n".format(folder,job))
                elif "BSUB -q" in fstr:
                    l = len(fstr)
                    sentence = fstr.split("_")[0]+"_2hr\n"
                    w.write(sentence)
                else:
                    w.write(fstr)
         
              else:
                if "#BSUB -J" in fstr:
                    w.write("#BSUB -J {}{}\n".format(folder,job))
                else:
                    w.write(fstr)
                    
        subprocess.call("bsub < batch.sh",shell=True)
 
def make_foldername(foldername):

  return "$HOME/"+foldername[22:len(foldername)]

def check_calculation(outcar_name,oszicar_name,foldername_):

  foldername = make_foldername(foldername_)

  # check queue is running
  q_end = True
  bjobs = subprocess.check_output("bjobs",shell=True,text=True)
  bjobs = bjobs.split("\n")
  pending = "not started"
  job_nums= []
  pendings = []
  for bjob in bjobs:
    if 'sung.w' in bjob:
      job_nums.append(bjob.split()[0])
      pendings.append(bjob.split()[2])

  for i in range(len(job_nums)):
    job_num = job_nums[i]
    pend    = pendings[i]
 # for job_num,pend in zip(job_nums,pendings):
    #print (job_num,pend)
    try:
      path0 = subprocess.check_output("bjobs -l "+job_num+" | grep CWD -9",shell=True,text=True).split()
    except:
      path0 = ['']
    path = ''
    for p in path0:
      path = path + p

    if foldername+'>' in path:
      if "PEND" in pend:
        pending = "pending"
      elif "RUN" in pend:
        pending = "running"
      q_end = False
 
  if os.path.isfile(outcar_name) == False:
    return q_end, pending, False, False,False,False
  else:
   with open(outcar_name,"r") as fi:
    outcar = fi.readlines()

   # check calculation is end
   if "Voluntary context" in outcar[-1]:
    calculation_end = True
   else:
    calculation_end = False
   # check if calculation is converged
   for fstr in outcar:
    if "NELM   =" in fstr:
      NELM = int(str(fstr.split()[2]).split(';')[0])
      break
    if "IBRION" in fstr:
      IBRION = int(str(fstr.split()[2])) 

   with open(oszicar_name,"r") as f:
    oszicar = f.readlines()

   maxc = 0
   ave_conv = 0
   max_step = 0
   divergence = 0

   for i in range(len(oszicar)):
    step = 0
    if "E0" in oszicar[i]:
      convergence = int(oszicar[i-1].split()[1])
      step        = int(oszicar[i].split()[0]) 
      ave_conv += convergence

      converge_complete = True

      if convergence >= NELM:
        divergence = 1
        converge_complete = False
        break

      if convergence >= maxc:
        maxc = convergence
        max_step = step

    else:
      converge_complete = False

#  if divergence == 0:

   for fstr in outcar:
    if "NSW    =" in fstr:
      NSW = int(str(fstr.split()[2]).split(';')[0])
      break

   if IBRION >= 1:
    if step >= NSW:
      ionic_step_end = False
    else:
      ionic_step_end = True

   if IBRION == 0:
    if step == NSW:
      ionic_step_end = True
    else:
      ionic_step_end = False

   return q_end, pending, calculation_end, converge_complete, ionic_step_end, True
