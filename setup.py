from setuptools import setup, find_namespace_packages

setup(
    name='modeperturber',
    version='0.0.1',
    author='Kang mingi',
    author_email='kangmg@korea.ac.kr',
    description='Normal mode like geometry perterber',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url='https://github.com/kangmg/ModePerturber',
    keywords=['chemistry','geometry','normal mode'],
    packages=find_namespace_packages(), 
    install_requires=[
        'ase',
        'torchani',
        'pointgroup',
        'pymsym',
        'git+https://github.com/kangmg/PySCF4ASE.git', # for gpu acceleration check https://github.com/kangmg/PySCF4ASE
        'git+https://github.com/isayevlab/AIMNet2.git@82dcc21d059243f7ee30fe0427c1b718def1e9cb', # aimnet2calc
        'git+https://github.com/Nilsgoe/mace_hessian.git@31caf70aea7642dd0783eb5129e967ca7352c971' # mace_hessian
    ],
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
)