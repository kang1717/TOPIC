from setuptools import find_packages, setup


install_requires = [
    'matplotlib<3.4.0,>=3.1.0',
    'scikit-learn<1.0',
    'numpy>=1.22',
    'scipy<1.6.0',
    'ase==3.22.0',
    'pyyaml',
    'cffi',
    'psutil',
    'tqdm',
    'braceexpand',
]

setup_requires = [
    'cffi',
]

setup(
    name='topic',
    version='0.1.0',
    description='TOPIC',
    author='Sungwoo Kang, Seungwoo Hwang',
    python_requires='>=3.6',
    packages=find_packages(include=['spinner', 'spinner*']),
    include_package_data=True,
    package_data={
        '': ['configure_default.yaml', 'INCAR_premelt', 'KPOINTS', '*.cpp', '*.h', 'params_*'],
    },
    entry_points={
        'console_scripts':[
            'spinner_auto_md = topic.auto_md.spinner_auto_md:main',
            'spinner_nnp_train = topic.nnp_train.initial_NNP_training:main',
            'configure_csp = topic.utils.configure_csp:main',
            'topic_csp = topic.topology_csp.topology_csp:main',
            #'spinner_dft_relax = spinner.spinner_dft_relax:main',
        ]
    },
    install_requires=install_requires,
    setup_requires=setup_requires,
    cffi_modules=[
        "topic/simple_nn/features/symmetry_function/libsymf_builder.py:ffibuilder",
        "topic/simple_nn/utils/libgdf_builder.py:ffibuilder",
    ],
)
