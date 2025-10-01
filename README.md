# TOPIC_refactoring
TOPIC(TOpology-based crystal structure Prediction of Inorganic solid electrolytes with Corner-sharing frameworks)

If you use TOPIC, please cite this paper: xxxx.

## Installation

### Virtual environment
TOPIC utilize not only TOPIC source but also many other packages such as randspg, LAMMPS, SIMPLE-NN, and AMP2.
So, we highly recommand to use virtual environment such as Anaconda.
After install Anaconda, create virtual environment for TOPIC and activate the environment.
```
  conda create -n TOPIC python=3.9
  conda activate TOPIC
```

### Download TOPIC
```
git clone https://github.com/MDIL-SNU/TOPIC.git
```

### Python requirements

TOPIC supports Python `3.6` or higher version. TOPIC utilizes Python modules in the following:

  - mpi4py (https://bitbucket.org/mpi4py/mpi4py/)
  - PyYAML (https://pypi.org/project/PyYAML/)
  - numpy (1.x.x version) (https://numpy.org/)
  - pybind11 (https://github.com/pybind/pybind11/)
  - pymatgen ()

You can download these modules by

```
  pip3 install mpi4py PyYAML numpy==1.26.4 pybind11 pymatgen
```

or directly downloading the source codes from the webpages.

Following packages are required to use SIMPLE-NN and AMP2

  - PyTorch (https://pytorch.org/)
  - spglib (https://github.com/spglib/spglib)


### C++ compilers

[TODO]
TOPIC supports CMAKE 3.20 or higher version and gcc version 7.3.1 or higer version.

### Python binding of randSpg and LAMMPS (important)
TOPIC utilizes python binding of randSpg (https://github.com/xtalopt/randSpg) and LAMMPS code (https://www.lammps.org). However, those codes are slightly modified to be incorporated with TOPIC, so you need to comiple with source codes provided in this page, not with the original ones.

### Install randSpg
To install randSpg, do

```
  cd /TOPIC-directory/randSpg-vspinner/
  mkdir build
  cd build
  cmake ..
  make -j3
```

to bind randSpg with python, do 

```
  cd /TOPIC-directory/randSpg-vspinner/python
  mkdir build
  cd build
  cmake ..
  make –j3
  cp pyrandspg.cpython* /directory-where-your-python-is/lib/python3/site-packages/
```

(FAQ) if error occur during cmake process, try

```
  cd /TOPIC-directory/randSpg-vspinner/python
  mkdir build
  cd build
  cmake .. -D pybind11_DIR=/VIRTUAL_ENV_DIR/lib/python3.9/site-packeges/pybind11/share/cmake/pybind11/ 
  make –j3
  cp pyrandspg.cpython* /directory-where-your-python-is/lib/python3/site-packages/
```

### Install LAMMPS
This is the most tricky part of the installation steps. You can install python-lammps by following steps below but it may not work depending on your environment. Please look into LAMMPS forum (https://www.lammps.org/forum.html) or LAMMPS manual Ch. 2 (https://docs.lammps.org/Python_run.html) for detailed discussion.

First, download stable version of LAMMPS from LAMMPS github (https://github.com/lammps/lammps.git)
We lastly tested on 2Aug2023 version.

```
  git clone -b stable https://github.com/lammps/lammps.git lammps-spinner
```

TOPIC provide two version of LAMMPS, Normal and SIMD. Install process is different for each version. Check your machine and follow the install process that match with your machine.

#### 1. Normal version
Copy pair potential code of SIMPLE-NN to LAMMPS src

```
  cp /TOPIC-directory/spinner/simple_nn/features/symmetry_function/pair_nn_simd.* /LAMMPS-directory/src/
  cp /TOPIC-directory/spinner/simple_nn/features/symmetry_function/symmetry_function.h /LAMMPS-directory/src/
```

```
  cd /LAMMPS-directory/
  cd src
  make yes-python
  make XXX mode=shlib
  make install-python
```

Here, XXX is the name of the make file (Makefile.XXX in src/MAKE directory). Note that LAMMPS have to be installed with serial version (not mpi version). The optimization is recommended for your system but default serial option is likely to be sufficient.

To check whether LAMMPS install in python, run the following code in python.

```
from lammps import lammps
lmp = lammps()
```

#### 2. SIMD version
If your machine support SIMD, we recommend to use SIMD version. It speeds up around 2 times faster than normal version.

```
  cp /TOPIC-directory/spinner/simple_nn/features/symmetry_function/SIMD/pair_nn_simd.* /LAMMPS-directory/src/
```

```
  cd /LAMMPS-directory/
  cp cmake/preset/oneapi.cmake cmake/preset/my_oneapi.cmake 
```

find the below line and add "-xAVX" tag in the cmake/preset/my_opeapi.cmake

```
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -xAVX" CACHE STRING "" FORCE)
```

add the bellow line in the cmake/preset/my_opeapi.cmake

```
set(ENV{VIRTUAL_ENV} "/YOUR_VIRTUAL_ENV_DIR/")
```

```
  mkdir build
  cd build
  cmake -D PKG_PYTHON=yes -D PKG_EXTRA-COMPUTE=yes -D PKG_EXTRA-PAIR=yes -D PKG_INTEL=yes \
  -D BUILD_MPI=no -D LAMMPS_MACHINE=simd_serial  -D INTEL_ARCH=cpu -D BUILD_SHARED_LIBS=yes  \
  -D  CMAKE_BUILD_TYPE=Release  -D CMAKE_INSTALL_PREFIX=/VIRTUAL_ENV_DIR/ \
  -C ../cmake/presets/nolib.cmake -C ../cmake/presets/my_oneapi.cmake ../cmake/
  cmake --build . --target install
  cmake --build . --target install-python
```

To check whether LAMMPS install in python, run the following code in python.

```
from lammps import lammps
lmp = lammps('simd_serial')
```

### Finalize installation

Other packages are automatically installed by running the bellow command.

```
  cd /TOPIC-directory/
  pip install .
```

## Usage

## 1. Running only TOPIC code (crystal structure prediction part)

Check the examples in the /TOPIC-directory/examples/csp directory.

To use TOPIC, 1 file (XXX.yaml) and 1 directories (input directory and src) are required.

### input file (XXX.yaml; XXX is named by the user)
Parameter list to control TOPIC code is listed in XXX.yaml. 
The simplest form of input.yaml is described below:
```YAML
# input.yaml
directory:
  input_path:        input_directory_name
cation_cn:
  P: 4
  Ti: 6
generation: 400
material:
  Li: 4
  O: 20
  P: 4
  Ti: 4
volume:   500.0
```

### input directory
In input directory (input_dir: in input file), LAMMPS potential file should be located. (Potential file have to be named potential.) Potential should be SIMPLE-NN format (https://github.com/MDIL-SNU/SIMPLE-NN).


### Running code

```
  mpirun -np core_number topic_csp input.yaml
```

## 2. Running whole steps in serial (from DFT MD to crystal structure prediction)

Check the examples in the /TOPIC-directory/examples/serial_run directory.

To use serial mode, 1 file (total.yaml) are required.

### input file (total.yaml)
Parameter list to control whole process code is listed in total.yaml. 
Check detail parameters in /TOPIC-directory/examples/serial_run/total.yaml.

### Running code

```
  # DFT melt-quench MD
  topic_auto_md -np core_number total.yaml

  # NNP training with SIMPLE-NN package
  topic_nnp_train total.yaml
```
