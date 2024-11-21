from setuptools import find_packages, setup


install_requires = [
    'matplotlib<3.4.0,>=3.1.0',
    'scikit-learn<1.0',
    'numpy>=1.22,<2.0.0',
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
    packages=find_packages(include=['topic', 'topic*']),
    include_package_data=True,
    package_data={
        '': ['configure_default.yaml', 'INCAR_premelt', 'KPOINTS', '*.cpp', '*.h', 'params_*', 'INCAR0', 'INCAR1'],
    },
    entry_points={
        'console_scripts':[
            'topic_auto_md = topic.auto_md.topic_auto_md:main',
            'topic_nnp_train = topic.nnp_train.initial_NNP_training:main',
            'configure_csp = topic.utils.configure_csp:main',
            'topic_csp = topic.topology_csp.topic_csp:main',
            'topic_post = topic.topology_csp.post_processing:main',
            'topic_dft = topic.dft_calculation.run:main',
            #'topic_dft_relax = topic.topic_dft_relax:main',
        ]
    },
    install_requires=install_requires,
    setup_requires=setup_requires,
    cffi_modules=[
        "topic/simple_nn/features/symmetry_function/libsymf_builder.py:ffibuilder",
        "topic/simple_nn/utils/libgdf_builder.py:ffibuilder",
    ],
)
