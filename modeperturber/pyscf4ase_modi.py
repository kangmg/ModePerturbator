import ase
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
import torch
import pyscf
import numpy as np

class PySCFCalculator(Calculator):
    '''
    Description
    ===========
    PySCF ASE Calculator for dft caculations
    '''

    '''
    ===============
    Units
    ---------------
    Energy  eV
    Force   eV/Ang
    ===============
    '''


    implemented_properties = ['energy', 'forces', 'hessian']

    default_parameters = {
        'charge': 0, # system charge
        'spin': 0, # (= nelec alpha-beta = 2S)
        'symmetry': False, # point group symmetry : e.g. Cs or C2v Ref. https://github.com/pyscf/pyscf/blob/master/examples/gto/13-symmetry.py
        'basis': 'def2-tzvp', # basis set
        'xc': 'wb97m-d3bj', # xc functional
        'density_fit': True, # resolution of identity approximation
        'auxbasis': 'auto', # basis for density_fit : e.g. 'cc-pvdz-jkfit' or 'auto'(pyscf sets automatically)
        'device': 'auto', # gpu, cpu
        'disp': None, # dispersion correction : d3bj d4 d3 etc.
        'nlc': 'auto', # non-local correlation. :  0, 'auto', nlc-functional
        # [Note]
        # Set nlc to 0 when D3 and D4 dispersion corrections are applied
        # https://github.com/pyscf/pyscf/blob/master/examples/dft/15-nlc_functionals.py
        # https://github.com/pyscf/pyscf/blob/master/examples/dft/16-dft_d3.py
        'max_cycle': 50, # max number of iterations
        'conv_tol': 1e-9, # converge threshold
        'verbose' : 4, # output log level
        'max_memory': 150000, # MB unit
        'chkfile': None, # chkpoint file contains MOs, orbital energies etc. : (str) e.g. './checkpoint/pyscf.chk'
        'output': None, # log file : (str) e.g. './output/pyscf_output.log'
        # TODO : conv_tol_grad, init_guess
    }

    def _set_device(self):
        # set device
        if self.parameters.device == 'auto':
            self.parameters.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        # import modules : dft
        if self.parameters.device == 'cpu':
            from pyscf import dft
            self.dft = dft
        elif self.parameters.device == 'gpu':
            from gpu4pyscf import dft
            self.dft = dft
        else:
            raise ValueError(f"Invalid device: {self.parameters.device}, 'gpu' or 'cpu' supported.")

    def __init__(self, restart=None, label='PySCF', **kwargs):
        super().__init__(restart=restart, label=label, **kwargs)
        self.mol = None
        self.mf = None

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        self._set_device()

        if self.atoms is None:
            raise ValueError("atoms object is not set.")

        if system_changes:
            self.results.clear()
            self.mol = None
            self.mf = None

        if self.mol is None:
            self._generate_molecule()

        if self.mf is None:
            self._run_dft()

        if 'energy' in properties and 'energy' not in self.results:
            self.results['energy'] = self.mf.e_tot * Ha

        if 'forces' in properties and 'forces' not in self.results:
            self.results['forces'] = self._calculate_forces() * Ha / Bohr

        if 'hessian' in properties and 'hessian' not in self.results:
            self.results['hessian'] = self._calculate_hessian() * Ha / (Bohr**2)

    def _generate_molecule(self):
        positions = self.atoms.get_positions()
        symbols = self.atoms.get_chemical_symbols()
        atom_str = "; ".join([f"{s} {p[0]} {p[1]} {p[2]}" for s, p in zip(symbols, positions)])
        self.mol = pyscf.M(atom=atom_str,
                         basis=self.parameters.basis,
                         charge=self.parameters.charge,
                         spin=self.parameters.spin,
                         symmetry=self.parameters.symmetry,
                         verbose = self.parameters.verbose,
                         max_memory=self.parameters.max_memory,
                         output=self.parameters.output,
                         unit='Angstrom')


    def _run_dft(self):
        self.mf = self.dft.RKS(self.mol)
        self.mf.xc = self.parameters.xc
        self.mf.max_cycle = self.parameters.max_cycle
        self.mf.conv_tol = self.parameters.conv_tol
        self.mf.disp = self.parameters.disp

        # checkpoint file : default is /tmp
        if self.parameters.chkfile:
            self.mf.chkfile = self.parameters.chkfile

        # non-local correlation
        if self.parameters.nlc != 'auto':
            self.mf.nlc = self.parameters.nlc

        # density_fit ( or resolution of identity (RI) approximation)
        if self.parameters.density_fit:
            if self.parameters.auxbasis == 'auto':
                self.mf.density_fit()
            elif self.parameters.auxbasis != 'auto':
                self.mf.density_fit(auxbasis=self.parameters.auxbasis)

        self.mf.kernel()

    def _calculate_forces(self):
        return -self.mf.nuc_grad_method().kernel()

    def _calculate_hessian(self):
        hessian = self.mf.Hessian().kernel()
        return hessian

    def get_hessian(self):
        """
        Calculate and return the Hessian matrix.

        Returns:
        --------
        hessian : numpy.ndarray
            The Hessian matrix with shape (3N, 3N) where N is the number of atoms.
            Units are in eV/Ang^2.
        """
        if 'hessian' not in self.results:
            self.calculate(properties=['hessian'])
        return self.results['hessian']