import os, sys, yaml
from topic.topology_csp.basic_tools import read_poscar_dict, direct2cartesian
import copy
from copy import deepcopy
import random
import lammps
from lammps import lammps
import math

########### read input.yaml file #############

input_file = str(sys.argv[1])
with open(input_file, 'r') as f:
    total_yaml = yaml.safe_load(f)

material    = total_yaml['material']
bond_dict   = total_yaml['distance_constraint']
second_dict = total_yaml['distance_constraint_2']

cat_info = {k: v for k, v in material.items() if k not in {'Li', 'O'}}
cat_info = sorted(cat_info.items(), key=lambda item: item[1], reverse=True)
frame_info = dict(cat_info) | dict([('O', material['O'])])

# bond distance
O_O     = bond_dict["O-O"]
m1_m1   = bond_dict[f"{cat_info[0][0]}-{cat_info[0][0]}"]
m2_m2   = bond_dict[f"{cat_info[1][0]}-{cat_info[1][0]}"]
if f"{cat_info[0][0]}-{cat_info[1][0]}" in bond_dict:
    m1_m2 = bond_dict[f"{cat_info[0][0]}-{cat_info[1][0]}"]
else:
    m1_m2 = bond_dict[f"{cat_info[1][0]}-{cat_info[0][0]}"]

# atom index for each elements
m1_start    = 1
m1_end      = cat_info[0][1]
m2_start    = cat_info[0][1] + 1
m2_end      = sum(dict(cat_info).values())
O_start     = sum(dict(cat_info).values()) + 1
O_end       = sum(frame_info.values())

bond_dict   = {key: value / (2**(1/6)) for key, value in bond_dict.items()}

##############################################

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

    O_start = sum(dict(cat_info).values()) + 1
    end = sum(frame_info.values())

    with open("in.in","w") as lfile:
        lfile.write("boundary p p p\n")
        lfile.write("units metal\n")
        lfile.write("atom_modify map yes\n")
        lfile.write("neighbor 2.0 bin\n")
        lfile.write("neigh_modify every 1 delay 0 check yes\n")
        lfile.write("\nread_data coo\n")
        lfile.write("\nmass * 1\n")
        lfile.write("\natom_modify sort 0 0.0\n")
        lfile.write("\nthermo 1\n")
        lfile.write("thermo_style custom step temp pe evdwl ecoul vol press spcpu\n")
        lfile.write("dump 1 all custom 10 out.xyz id type xu yu zu\n")
        lfile.write("dump_modify 1 sort id\n")
        lfile.write("\npair_style hybrid/overlay lj/cut 6.0 harmonic/cut\n")
        lfile.write("\npair_coeff   *     *   harmonic/cut       0.0  0.0\n")
        lfile.write(f"pair_coeff {O_start}*{end} {O_start}*{end} harmonic/cut       0.01 2.0\n")
        lfile.write("pair_coeff * * lj/cut         0.0 0.0   0.0\n")
        lfile.write("\nminimize 0.0 2.0e-3 1000 10000000\n")
        lfile.write("\npair_coeff   *     *   harmonic/cut       0.0  0.0\n")

    with open("in.nnp","w") as lfile:
        lfile.write("boundary p p p\n")
        lfile.write("units metal\n")
        lfile.write("atom_modify map yes\n")
        lfile.write("neighbor 2.0 bin\n")
        lfile.write("neigh_modify every 1 delay 0 check yes\n")
        lfile.write("\nread_data coo_out2\n")
        lfile.write("\npair_style nn/intel\n")
        lfile.write(f"pair_coeff * * potentialshort {' '.join(list(frame_info.keys()))}\n")
        lfile.write("\nmass * 1\n")
        lfile.write("\natom_modify sort 0 0.0\n")
        lfile.write("\nthermo 1\n")
        lfile.write("thermo_style custom step temp pe evdwl ecoul vol press spcpu\n")
        lfile.write("\n\ndump 1 all custom 10 out.xyz id type xu yu zu\n")
        lfile.write("dump_modify 1 sort id\n")
        lfile.write("\n\nminimize 0.0 2.0e-1 1000  10000000\n")
        lfile.write("\nfix F1 all box/relax tri 0.0 vmax 0.00001\n")
        lfile.write("minimize 0.0 5.0e-2 1000 10000000\n")
        lfile.write("\nunfix F1\n")
        lfile.write("fix F2 all box/relax tri 0.0 vmax 0.001\n")
        lfile.write("minimize 0.0 3.0e-2 1000 10000000\n")
        lfile.write("\nunfix F2\n")
        lfile.write("fix F3 all box/relax tri 0.0 vmax 0.01\n")
        lfile.write("minimize 0.0 2.0e-2 1000 10000000\n")
        lfile.write("\n\nwrite_data coo_nnp\n")

    with open("in.nnplong","w") as lfile:
        lfile.write("boundary p p p\n")
        lfile.write("units metal\n")
        lfile.write("atom_modify map yes\n")
        lfile.write("neighbor 2.0 bin\n")
        lfile.write("neigh_modify every 1 delay 0 check yes\n")
        lfile.write("\nread_data coo_out3")
        lfile.write("\npair_style nn/intel\n")
        lfile.write(f"pair_coeff * * potentiallong {' '.join(list(frame_info.keys()))}\n")
        lfile.write("\nmass * 1\n")
        lfile.write("\natom_modify sort 0 0.0\n")
        lfile.write("\nthermo 1\n")
        lfile.write("thermo_style custom step temp pe evdwl ecoul vol press spcpu\n")
        lfile.write("\n\nminimize 0.0 2.0e-1 100  10000000\n")
        lfile.write("\nfix F1 all box/relax tri 0.0 vmax 0.00001\n")
        lfile.write("minimize 0.0 5.0e-4 100 10000000\n")
        lfile.write("\nunfix F1\n")
        lfile.write("fix F2 all box/relax tri 0.0 vmax 0.001\n")
        lfile.write("minimize 0.0 4.0e-4 100 10000000\n")
        lfile.write("\nunfix F2\n")
        lfile.write("fix F3 all box/relax tri 0.0 vmax 0.01\n")
        lfile.write("minimize 0.0 3.0e-4 100 10000000\n")
        lfile.write("\n\nwrite_data coo_nnp2\n")

    with open("in.out","w") as lfile:
        lfile.write(f"pair_coeff {m1_start}*{m1_end}     {m1_start}*{m1_end}      harmonic/cut 0.2 {m1_m1}\n")
        lfile.write(f"pair_coeff {m1_start}*{m1_end}     {m2_start}*{m2_end}     harmonic/cut 0.2 {m1_m2}\n")
        lfile.write(f"pair_coeff {m2_start}*{m2_end}    {m2_start}*{m2_end}     harmonic/cut 0.2 {m2_m2}\n")
        lfile.write(f"pair_coeff {O_start}*{O_end}   {O_start}*{O_end}    harmonic/cut 0.2 {O_O}\n")
####  lfile.write("\n\nminimize 0.0 2.0e-1 1000 10000000\n")
####  lfile.write("\nfix F1 all box/relax tri 0.0 vmax 0.00001\n")
####  lfile.write("minimize 0.0 2.0e-3 2000 10000000\n")
####  lfile.write("\nunfix F1\n")
####  lfile.write("fix F2 all box/relax tri 0.0 vmax 0.001\n")
####  lfile.write("minimize 0.0 2.0e-3 2000 10000000\n")
####  lfile.write("\nunfix F2\n")
####  lfile.write("fix F3 all box/relax tri 0.0 vmax 0.01\n")
####  lfile.write("minimize 0.0 2.0e-3 2000 10000000\n")
####  lfile.write("\n\nwrite_data coo_out\n")


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
  lmp = lammps('simd_serial')
  #lmp = lammps('simd_serial', cmdargs=["-log", "none", "-screen", os.devnull,  "-nocite"])
  try:
      lmp.file(filename)  
      lmp.command("variable e equal etotal")
      lmp.command("variable v equal vol")
      e1 = lmp.extract_variable("e",'all')
      v1 = lmp.extract_variable("v",'all')
      return e1,v1
  except:
      return 0,0

def run_lj_lammps(filename):
    lmp = lammps('simd_serial')
    lmp.file(filename)  
    lmp.command("minimize 0.0 2.0e-1 1000 10000000")
    lmp.command("fix F1 all box/relax tri 0.0 vmax 0.00001")
    lmp.command("minimize 0.0 2.0e-3 2000 10000000")
    lmp.command("variable p equal press")
    p = lmp.extract_variable("p",'all')
    if p > 10000:
        return 0,0
    lmp.command("unfix F1")
    lmp.command("fix F2 all box/relax tri 0.0 vmax 0.001")
    i=0
    if i < 11:
        i+=1
        lmp.command("minimize 0.0 2.0e-3 200 10000000")
        lmp.command("variable p equal press")
        p = lmp.extract_variable("p",'all')
        if abs(p) > 10000:
            return 0,0
    #lmp.command("unfix F2")
    #lmp.command("fix F3 all box/relax tri 0.0 vmax 0.01")
    #i=0
    #if i < 11:
    #    i+=1
    #    lmp.command("minimize 0.0 2.0e-3 200 10000000")
    #    lmp.command("variable p equal press")
    #    p = lmp.extract_variable("p",'all')
    #    if abs(p) > 10000:
    #        return 0,0
    lmp.command("variable e equal etotal")
    lmp.command("variable v equal vol")
    e1 = lmp.extract_variable("e",'all')
    v1 = lmp.extract_variable("v",'all')
    lmp.command("write_data coo_out") 

    return e1,v1
