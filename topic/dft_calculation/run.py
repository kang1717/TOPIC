import os
import sys
import shutil
import random
import subprocess
import numpy as np
import math
import yaml


def get_abs_path(*paths):
    path = os.path.join(*paths)
    if not path.startswith('/'):
        path = '/' + path
    return path

def make_potcar(pot_dir):

    with open("POSCAR","r") as f:
        for i in range(5):
            f.readline()
        atoms = f.readline().split()

    fw = open("POTCAR","w")

    for atom in atoms:
        if atom == 'Cs':
            with open(pot_dir+'/'+atom+"_sv_GW/POTCAR","r") as f:
                for fstr in f.readlines():
                    fw.write(fstr)
        elif atom == 'Ca':
            with open(pot_dir+'/'+atom+"_sv/POTCAR","r") as f:
                for fstr in f.readlines():
                    fw.write(fstr)
        elif atom in ['Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']:
            with open(pot_dir+'/'+atom+"_3/POTCAR","r") as f:
                for fstr in f.readlines():
                    fw.write(fstr)
        elif os.path.isdir(pot_dir+"/"+atom):
            with open(pot_dir+'/'+atom+"/POTCAR","r") as f:
                for fstr in f.readlines():
                    fw.write(fstr)
        elif os.path.isdir(pot_dir+"/"+atom+'_pv'):
            with open(pot_dir+'/'+atom+"_pv/POTCAR","r") as f:
                for fstr in f.readlines():
                    fw.write(fstr)
        elif os.path.isdir(pot_dir+"/"+atom+'_sv'):
            with open(pot_dir+'/'+atom+"_sv/POTCAR","r") as f:
                for fstr in f.readlines():
                    fw.write(fstr)

    fw.close()

def input_yaml(input_file):
    with open(input_file,'r') as inp:
        inp_yaml = yaml.safe_load(inp)

    tot_atom_num = 0
    for atom in inp_yaml['material']:
        tot_atom_num += inp_yaml['material'][atom]
    inp_yaml['tot_atom_num'] = int(tot_atom_num) 

    return inp_yaml

def check_if_there_is_missing_input(inp_yaml):

    # requirements are input_dir, output_dir, structure, material, initial volume

    if   'output_dir' not in inp_yaml:
        return "output_dir: is missing in input.yaml"

    elif 'input_dir'  not in inp_yaml:
        return "input_dir: is missing in input.yaml"

    elif 'structure' not in inp_yaml:
        return "structure: generation: is missing in input.yaml"

    elif 'generation' not in inp_yaml['structure']:
        return "structure: generation: is missing in input.yaml"

    elif 'material' not in inp_yaml:
        return "material: generation: is missing in input.yaml"

    elif 'initial_volume' not in inp_yaml:
        return "initial_volume: is missing in input.yaml"

    else:
        return ""


def default_inputs(inp_yaml):
    tot_atom_num = inp_yaml['tot_atom_num']

    # operator
    if 'operator' not in inp_yaml:
        inp_yaml['operator'] = {}
        inp_yaml['operator']['random'] = 0.7
        inp_yaml['operator']['crossover'] = 0.0
        inp_yaml['operator']['con_permutation'] = 0.0
        inp_yaml['operator']['all_permutation'] = 0.2
        inp_yaml['operator']['latticemutation'] = 0.1

    else:
        if 'random' not in inp_yaml['operator']:
            inp_yaml['operator']['random'] = 0.0
        if 'crossover' not in inp_yaml['operator']:
            inp_yaml['operator']['crossover'] = 0.0
        if 'con_permutation' not in inp_yaml['operator']:
            inp_yaml['operator']['con_permutation'] = 0.0
        if 'all_permutation' not in inp_yaml['operator']:
            inp_yaml['operator']['all_permutation'] = 0.0
        if 'latticemutation' not in inp_yaml['operator']:
            inp_yaml['operator']['latticemutation'] = 0.0

    # structure
    if 'i_population' not in inp_yaml['structure']:
        inp_yaml['structure']['i_population'] = inp_yaml['tot_atom_num']*2

    if 'population' not in inp_yaml['structure']:
        inp_yaml['structure']['population'] = inp_yaml['tot_atom_num']*2

    if 'num_of_best' not in inp_yaml['structure']:
        inp_yaml['structure']['num_of_best'] = 0

    if 're-relax_best' not in inp_yaml['structure']:
        inp_yaml['structure']['re-relax_best'] = True

    # energy_critetria:
    if 'energy_criteria' not in inp_yaml:
        inp_yaml['energy_criteria'] = {}

    if 'energy_cut_for_inheriting_structures' not in inp_yaml['energy_criteria']:
        inp_yaml['energy_criteria']['energy_cut_for_inheriting_structures'] = 0.10

    if 'energy_cut_for_best_structures' not in inp_yaml['energy_criteria']:
        inp_yaml['energy_criteria']['energy_cut_for_best_structures'] = 0.05

    if 'energy_cut_for_further_relax' not in inp_yaml['energy_criteria']:
        inp_yaml['energy_criteria']['energy_cut_for_further_relax'] = 100.0

    # crossover condition
    if 'crossover_condition' not in inp_yaml:
        inp_yaml['crossover_condition'] = {}

    if 'num_of_grid_for_cut' not in inp_yaml['crossover_condition']:
        inp_yaml['crossover_condition']['num_of_grid_for_cut'] = 10

    if 'energy_range_for_cut_select' not in inp_yaml['crossover_condition']:
        inp_yaml['crossover_condition']['energy_range_for_cut_select'] = 10

    if 'grid_for_shift' not in inp_yaml['crossover_condition']:
        inp_yaml['crossover_condition']['grid_for_shift'] = 3

    if 'iteration_for_add_atoms' not in inp_yaml['crossover_condition']:
        inp_yaml['crossover_condition']['iteration_for_add_atoms'] = 50

    # random_condition
    if 'random_condition' not in inp_yaml:
        inp_yaml['random_condition'] = {}

    if 'force_general_Wyckoff_site' not in inp_yaml['random_condition']:
        inp_yaml['random_condition']['force_general_Wyckoff_site'] = False
        
    if 'maximum_attempts_for_one_space_group_and_volume' not in inp_yaml['random_condition']:
        inp_yaml['random_condition']['maximum_attempts_for_one_space_group_and_volume'] = 100

    if 'scale_factor' not in inp_yaml['random_condition']:
        inp_yaml['random_condition']['scale_factor'] = 1.0

    if 'sublattice_generation' not in inp_yaml['random_condition']:
        inp_yaml['random_condition']['sublattice_generation'] = 0.0

    if 'max_sub_lattice' not in inp_yaml['random_condition']:
        inp_yaml['random_condition']['max_sub_lattice'] = 2

    # relax condition
    if 'relax_condition' not in inp_yaml:
        inp_yaml['relax_condition'] = {}

    if 'relax_iteration' not in inp_yaml['relax_condition']:
        inp_yaml['relax_condition']['relax_iteration'] = 5
    if 'method_of_first_relax' not in inp_yaml['relax_condition']:
        inp_yaml['relax_condition']['method_of_first_relax'] = 'cg'
    if 'further_calculate_with_accurate_potential' not in inp_yaml['relax_condition']:
        inp_yaml['relax_condition']['further_calculate_with_accurate_potential'] = False

    # distance constraint
    if 'distance_constraint' not in inp_yaml:
        inp_yaml['distance_constraint'] = {}

    for atom1 in inp_yaml['material']:
        for atom2 in inp_yaml['material']:

            if (atom1+'-'+atom2 not in inp_yaml['distance_constraint']) and (atom2+'-'+atom1 not in inp_yaml['distance_constraint']):
                inp_yaml['distance_constraint'][atom1+'-'+atom2] = 0.7

    # vacuum constraint
    if 'vacuum_constraint' not in inp_yaml:
        inp_yaml['vacuum_constraint'] = {}

    if 'apply_vacuum_constraint' not in inp_yaml['vacuum_constraint']:
        inp_yaml['vacuum_constraint']['apply_vacuum_constraint'] = True

    if 'maximum_vacuum_length' not in inp_yaml['vacuum_constraint']:
        inp_yaml['vacuum_constraint']['maximum_vacuum_length'] = 10.0

    if 'grid' not in inp_yaml['vacuum_constraint']:
        inp_yaml['vacuum_constraint']['grid'] = 1.0

    # similarity metric
    if 'similarity_metric' not in inp_yaml:
        inp_yaml['similarity_metric'] = {}

    if 'type' not in inp_yaml['similarity_metric']:
        inp_yaml['similarity_metric']['type'] = 'pRDF'

    if 'limit' not in inp_yaml['similarity_metric']:
        inp_yaml['similarity_metric']['limit'] = min(4.0/float(tot_atom_num),0.2)

    if 'volume_cut' not in inp_yaml['similarity_metric']:
        inp_yaml['similarity_metric']['volume_cut'] = 0.1

    if 'energy_cut' not in inp_yaml['similarity_metric']:
        inp_yaml['similarity_metric']['energy_cut'] = 0.005

    if 'gaussian_dist' not in inp_yaml['similarity_metric']:
        inp_yaml['similarity_metric']['gaussian_dist'] = 0.3

    if 'rdf_grid' not in inp_yaml['similarity_metric']:
        inp_yaml['similarity_metric']['rdf_grid'] = 250

    # antiseed
    if 'antiseed' not in inp_yaml:
        inp_yaml['antiseed'] = {}

    if 'activate_antiseed' not in inp_yaml['antiseed']:
        inp_yaml['antiseed']['activate_antiseed'] = False

    if 'gaussian_width' not in inp_yaml['antiseed']:
        inp_yaml['antiseed']['gaussian_width'] = 0.2

    if 'selection_gaussian' not in inp_yaml['antiseed']:
        inp_yaml['antiseed']['selection_gaussian'] = 0.1

    if 'selection_fraction' not in inp_yaml['antiseed']:
        inp_yaml['antiseed']['selection_fraction'] = 0.5

    # continue
    if 'continue' not in inp_yaml:
        inp_yaml['continue'] = {}
        inp_yaml['continue']['continue_num'] = 0

    return inp_yaml


def make_tolerance_matrix(atomnamelist, inp_file):
    
    tolerance_matrix = [[0.0 for i in range(len(atomnamelist))] for j in range(len(atomnamelist))]

    for atom1num in range(len(atomnamelist)):
        for atom2num in range(len(atomnamelist)): 
            atom1 = atomnamelist[atom1num]
            atom2 = atomnamelist[atom2num]
            pair1 = atom1 + "-" + atom2
            pair2 = atom2 + "-" + atom1

            if pair1 in inp_file['distance_constraint']:
                tolerance_matrix[atom1num][atom2num] = inp_file['distance_constraint'][pair1]
                tolerance_matrix[atom2num][atom1num] = inp_file['distance_constraint'][pair1]
            elif pair2 in inp_file['distance_constraint']: 
                tolerance_matrix[atom2num][atom1num] = inp_file['distance_constraint'][pair2]
                tolerance_matrix[atom1num][atom2num] = inp_file['distance_constraint'][pair2]
                
    return tolerance_matrix    

def make_kpt(k_spacing):
    with open("POSCAR","r") as f:
        f.readline()
        f.readline()
        latt = []
        latt.append(list(map(float,f.readline().split()[0:3])))
        latt.append(list(map(float,f.readline().split()[0:3])))
        latt.append(list(map(float,f.readline().split()[0:3])))
        latt = np.array(latt)

        k0 = math.ceil(1.0/(np.linalg.norm(latt[0])*k_spacing))
        k1 = math.ceil(1.0/(np.linalg.norm(latt[1])*k_spacing))
        k2 = math.ceil(1.0/(np.linalg.norm(latt[2])*k_spacing))

    with open("KPOINTS","w") as f:
        f.write("Auto k-point\n")
        f.write(" 0\n")
        f.write("Monk-horst\n") 
        f.write("%d  %d  %d\n"%(k0,k1,k2))
        f.write("0 0 0")

    return [k0, k1, k2]

def main():
    input_name  = sys.argv[1]
    mpi_command = sys.argv[2]
    NPROC       = sys.argv[3]

    inp_file = input_yaml(input_name)

    #src_dir = inp_file['DFT_setting']['src_dir']
    src_dir = os.path.dirname(os.path.abspath(__file__))
    vasp_gam = inp_file['DFT_setting']['vasp_gam']
    vasp_std = inp_file['DFT_setting']['vasp_std']
    pot_dir  = inp_file['DFT_setting']['potential_directory']
    k_spacing = inp_file['DFT_setting']['k-spacing']

    cwd=os.getcwd()
    os.makedirs("one_shot", exist_ok=True)
    os.chdir("one_shot")

    # make k-point module
    shutil.copy(get_abs_path(src_dir, 'INCAR1'), './INCAR')
    #edit_INCAR('./INCAR', {'SYSTEM':material, 'NSW': nsw})
    #shutil.copy('../INCAR1', 'INCAR')

    # run vasp with ISYM = 1
    for str_i, n in enumerate(os.listdir('../final_poscars')):
        tmp = n.split('_')
        #shutil.copy('../final_poscars/'+n, 'POSCAR_%s'%i_struct)
        shutil.copy('../final_poscars/'+n, 'POSCAR')
        kptarray = make_kpt(k_spacing)
        #shutil.copy("KPOINTS","KPOINTS"+str(i_struct))

        # make POTCAR 
        if str_i == 0:
            make_potcar(pot_dir)

        # run VASP
        if kptarray == [1,1,1]:
            with open("stdout.x_"+n,"w") as f:
                subprocess.call([mpi_command,'-np',NPROC,vasp_gam],stdout=f)

        else:
            with open("stdout.x_"+n,"w") as f:
                subprocess.call([mpi_command,'-np',NPROC,vasp_std],stdout=f)

        os.rename('OUTCAR','OUTCAR_%s_%s'%(tmp[1], tmp[2]))

    shutil.copy(get_abs_path(src_dir, 'INCAR0'), './INCAR')
    #shutil.copy('../INCAR0', 'INCAR')
    vasp_E = []
    for str_i, n in enumerate(os.listdir('../final_poscars')):
        tmp = n.split('_')
        try:
            check = subprocess.check_output(['grep', "free  ", 'OUTCAR_%s_%s'%(tmp[1], tmp[2])])
        except:
            check = ''

        if check =='':
            os.remove('OUTCAR_%s_%s'%(tmp[1], tmp[2]))

            shutil.copy('../final_poscars/'+n, 'POSCAR')
            kptarray = make_kpt(k_spacing)

            if kptarray == [1,1,1]:
                with open("stdout.x_"+n,"w") as f:
                    subprocess.call([mpi_command,'-np',NPROC,vasp_gam],stdout=f)

            else:
                with open("stdout.x_"+n,"w") as f:
                    subprocess.call([mpi_command,'-np',NPROC,vasp_std],stdout=f)

            os.rename('OUTCAR','OUTCAR_%s_%s'%(tmp[1], tmp[2]))
            #freeE = float(subprocess.check_output(['grep',"free  ",'OUTCAR'+str(i)]).split()[4])
            #vasp_E.append(freeE)
            with open('OUTCAR_%s_%s'%(tmp[1], tmp[2])) as f:
                lines = f.readlines()
                key = 0
                for j in range(len(lines)-1, 0, -1):
                    if 'free  ' in lines[j]:
                        vasp_E.append(float(lines[j].split()[4]))
                        key = 1
                        break
                if key == 0:
                    vasp_E.append(0)
        else:
            vasp_E.append((n, float(check.split()[4])))

    with open('DFT_results', 'w') as s:
        for n in vasp_E:
            s.write('{:15} {:9.3f}\n'.format(n[0], n[1]))
if __name__ == '__main__':
    main()
