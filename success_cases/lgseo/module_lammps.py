import os,sys
sys.path.append("/home/sung.w.kang/utill")
from basic_tools import *
import copy
from copy import deepcopy
import random
import lammps
from lammps import lammps

bond_dict  = {}
bond_dict['Ga-O'] = 1.94/(2**(1/6))
bond_dict['Se-O'] = 1.76/(2**(1/6))
bond_dict['O-Ga'] = 1.94/(2**(1/6))
bond_dict['O-Se'] = 1.76/(2**(1/6))

second_dict  = {}
second_dict['Ga-O']  = 3.86*0.9
second_dict['Se-O']  = 2.72*0.9
second_dict['O-Ga']  = 3.86*0.9
second_dict['O-Se']  = 2.72*0.9


pi=3.14159265358979
rad2deg=180.0/pi
 
def asin(a):
        return math.atan2(a,math.sqrt(1.0-a*a))
def acos(a):
        return pi/2.0-asin(a)
def norm(x):
        return (math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]))
def dotprod (x,y):
        return ( x[0]*y[0] + x[1]*y[1] + x[2]*y[2] )
def angle (v1,v2):
        myacos = dotprod(v1,v2)/norm(v1)/norm(v2)
        if myacos>1.0 :
            myacos = 1.0
        if myacos<-1.0:
            myacos = -1.0
        return(acos(myacos)*180.0/3.14159265358979)

def lammps_write(poscar_name,cation_dict):

 pair_lists = []
 pos = read_poscar_dict(poscar_name)

 lfile  = open("in.add1","w")

 fixnum = 0

 with open("in.add1","w") as lfile:
  for cation in cation_dict:
   c_dict = cation_dict[cation]

   for anion in c_dict:

    if [anion,cation] not in pair_lists and [cation,anion] not in pair_lists:

     bond = bond_dict[pos['atomarray'][cation]+'-'+pos['atomarray'][anion]]

     fixnum += 1

     lfile.write(f"pair_coeff {cation+1} {anion+1} lj/cut 3.0 {bond} 6.0\n") 

     pair_lists.append([cation,anion])


 fixnum = 0
 with open("in.add2","w") as lfile2:
  for i,cation in enumerate(sorted(cation_dict.keys())):

   for j,cation_other in enumerate(sorted(cation_dict.keys())):

    if i!=j:

      c_other_dict = cation_dict[cation_other]

      for anion_other in c_other_dict:
    
        if [anion_other,cation] not in pair_lists and [cation,anion_other] not in pair_lists:

         bond = second_dict[pos['atomarray'][cation]+'-'+pos['atomarray'][anion_other]]
         
         fixnum += 1

         lfile2.write(f"pair_coeff {cation+1} {anion_other+1} harmonic/cut 0.2 {bond} \n") 

         pair_lists.append([cation,anion_other])
 '''
 for i,cation in enumerate(sorted(cation_dict.keys())):
  for j,cation_other in enumerate(sorted(cation_dict.keys())):

    if i<j:

      c_dict = cation_dict[cation]
      c_other_dict = cation_dict[cation_other]

      for anion in c_dict:

        for anion_other in c_other_dict:

          if c_dict != c_other_dict:

            if [anion_other,anion] not in pair_lists and [anion,anion_other] not in pair_lists:

             fixnum += 1

             lfile.write(f"pair_coeff {anion+1} {anion_other+1} harmonic/cut 0.2 2.0\n") 

             pair_lists.append([anion,anion_other])
 '''
 file_names = ['in.in', 'in.add1', 'in.add2','in.out']
 output_file_name = 'in.all'

 with open(output_file_name, 'w') as output_file:
   for file_name in file_names:
     with open(file_name, 'r') as input_file:
       output_file.write(input_file.read())
       output_file.write('\n')

def POSCAR2cooall(filename,output='coo'):
 
    pos  = read_poscar_dict(filename) 
    if pos['cartesian'] == False:
      coor = direct2cartesian(pos['latt'],pos['coor'])
    else:
      coor = deepcopy(pos['coor'])
    latt = deepcopy(pos['latt'])

    # convert into lammps format
    p_a= math.sqrt(latt[0][0]**2.0 + latt[0][1]**2.0 + latt[0][2]**2.0);
    p_b= math.sqrt(latt[1][0]**2.0 + latt[1][1]**2.0 + latt[1][2]**2.0);
    p_c= math.sqrt(latt[2][0]**2.0 + latt[2][1]**2.0 + latt[2][2]**2.0);
    alpha=  angle(latt[1],latt[2]);  beta=  angle(latt[0],latt[2]); gamma=  angle(latt[0],latt[1]);  # Angles in degree
    alphar= alpha/rad2deg; betar= beta/rad2deg; gammar= gamma/rad2deg; # Angles in radians

    lx=   p_a
    p_xy= p_b * math.cos(gammar)
    p_xz= p_c * math.cos(betar)
    ly=   math.sqrt(p_b**2.0 - p_xy**2.0)
    p_yz= (p_b*p_c*math.cos(alphar)-p_xy*p_xz)/(ly)
    lz=   math.sqrt(p_c**2.0 - p_xz**2.0 - p_yz**2.0)

    atomnumlist = deepcopy(pos['numarray'])

    # print
    atomtype = []
    for i in range(len(atomnumlist)):
        atomtype += [i+1]*atomnumlist[i]
    latt = [lx,ly,lz,p_xy,p_xz,p_yz]

    with open(output,"w") as fw:
        fw.write(" POSCAR to lmp\n")
        fw.write("\n")
        fw.write(str(len(atomnumlist))+" atoms\n")
        fw.write(str(len(atomnumlist))+" atom types\n")
        fw.write("\n")
        fw.write("0.000000 %10.6f xlo xhi\n"%(lx))
        fw.write("0.000000 %10.6f ylo yhi\n"%(ly))
        fw.write("0.000000 %10.6f zlo zhi\n"%(lz))
        fw.write("\n")
        fw.write("%10.6f %10.6f %10.6f  xy xz yz\n"%(p_xy, p_xz, p_yz))
        fw.write("\n")
        fw.write("Atoms\n")
        fw.write("\n")
        for i in range(sum(pos['numlist'])):
            fw.write(" %d %d  %10.6f %10.6f %10.6f\n"%(i+1, i+1, coor[i][0], coor[i][1], coor[i][2]))

def POSCAR2coo(filename,output='coo'):
 
    pos  = read_poscar_dict(filename) 
    if pos['cartesian'] == False:
      coor = direct2cartesian(pos['latt'],pos['coor'])
    else:
      coor = deepcopy(pos['coor'])
    latt = deepcopy(pos['latt'])

    # convert into lammps format
    p_a= math.sqrt(latt[0][0]**2.0 + latt[0][1]**2.0 + latt[0][2]**2.0);
    p_b= math.sqrt(latt[1][0]**2.0 + latt[1][1]**2.0 + latt[1][2]**2.0);
    p_c= math.sqrt(latt[2][0]**2.0 + latt[2][1]**2.0 + latt[2][2]**2.0);
    alpha=  angle(latt[1],latt[2]);  beta=  angle(latt[0],latt[2]); gamma=  angle(latt[0],latt[1]);  # Angles in degree
    alphar= alpha/rad2deg; betar= beta/rad2deg; gammar= gamma/rad2deg; # Angles in radians

    lx=   p_a
    p_xy= p_b * math.cos(gammar)
    p_xz= p_c * math.cos(betar)
    ly=   math.sqrt(p_b**2.0 - p_xy**2.0)
    p_yz= (p_b*p_c*math.cos(alphar)-p_xy*p_xz)/(ly)
    lz=   math.sqrt(p_c**2.0 - p_xz**2.0 - p_yz**2.0)

    atomnumlist = deepcopy(pos['numarray'])

    # print
    atomtype = []
    for i in range(len(atomnumlist)):
        atomtype += [i+1]*atomnumlist[i]
    latt = [lx,ly,lz,p_xy,p_xz,p_yz]

    with open(output,"w") as fw:
        fw.write(" POSCAR to lmp\n")
        fw.write("\n")
        fw.write(str(len(atomnumlist))+" atoms\n")
        fw.write(str(len(pos['atomlist']))+" atom types\n")
        fw.write("\n")
        fw.write("0.000000 %10.6f xlo xhi\n"%(lx))
        fw.write("0.000000 %10.6f ylo yhi\n"%(ly))
        fw.write("0.000000 %10.6f zlo zhi\n"%(lz))
        fw.write("\n")
        fw.write("%10.6f %10.6f %10.6f  xy xz yz\n"%(p_xy, p_xz, p_yz))
        fw.write("\n")
        fw.write("Atoms\n")
        fw.write("\n")
        for i in range(sum(pos['numlist'])):
            fw.write(" %d %d  %10.6f %10.6f %10.6f\n"%(i+1, pos['numarray'][i]+1, coor[i][0], coor[i][1], coor[i][2]))

def run_lammps(filename):

  lmp = lammps(cmdargs=["-log", "none", "-screen", os.devnull,  "-nocite"])
  #lmp = lammps()
  try:
    lmp.file(filename)  
    lmp.command("variable e equal etotal")
    lmp.command("variable v equal vol")
    e1 = lmp.extract_variable("e",'all')
    v1 = lmp.extract_variable("v",'all')
    #lmp.command("write_data coo_out") 
    return e1,v1
  except:
    return 0,0
  finally:
    # Ensure that the LAMMPS object is properly disposed of
    lmp.close()

 
#from module_pair import *
#cation_dict,anion_dict = get_pair('POSCAR')
#lammps_write('POSCAR',cation_dict,anion_dict)

#POSCAR2coo("CONTCAR","coo_out2")
