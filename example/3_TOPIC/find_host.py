import pymatgen
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
import sys, os
import ase.io


if len(sys.argv) < 2:
    print('Run with python3 find_host.py POSCAR_DIR')
    exit()


sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False, scale=True, attempt_supercell=False)

file1 = '../ICSD.cif'
if 'POSCAR_ICSD' not in os.listdir('..'):
    d = ase.io.read('../ICSD.cif')
    ase.io.write('../POSCAR_ICSD', d, format='vasp')
file1 = '../POSCAR_ICSD'

if 'POSCAR' in file1:
    poscar1 = Poscar.from_file(file1)
    structure1 = poscar1.structure
elif '.cif' in file1:
    structure1 = CifParser(file1).parse_structures(primitive=False)[0]
li_idx1 = [i for i, site in enumerate(structure1) if site.species_string == 'Li']
structure1.remove_sites(li_idx1)

dirs = sys.argv[1]
tot = 0
for n in os.listdir(dirs):
    file2 = dirs+'/'+n
    poscar2 = Poscar.from_file(file2)
    structure2 = poscar2.structure

    li_idx2 = [i for i, site in enumerate(structure2) if site.species_string == 'Li']
    structure2.remove_sites(li_idx2)

    similarity = sm.fit(structure1, structure2)
    rms_dist = sm.get_rms_dist(structure1, structure2)

    if similarity:
        print(n, similarity, rms_dist)
        tot += 1
if tot == 0:
    print('No structure matched with reference structure')
