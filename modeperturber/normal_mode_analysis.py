_cached_models = {}

import torchani
import torch
import ase
import pymsym
import os
from torchani.utils import vibrational_analysis, get_atomic_masses, hessian
from mace.calculators import mace_off
from aimnet2calc import AIMNet2Calculator
from ase.io import write, read
from pointgroup import PointGroup
from pyscf4ase_modi import PySCFCalculator


def ani_hessian(atoms:ase.Atoms, model:str='ANI2x'):
    """
    Get Hessian using TorchANI calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure for which the Hessian matrix will be calculated.
    model : str, optional
        The TorchANI model to use. Options are ['ANI2x', 'ANI1x', 'ANI1ccx'].
        Default is 'ANI2x'.

    Returns
    -------
    hess : torch.Tensor
        The Hessian matrix as a torch tensor.
    """

    def _atoms2ani(atoms:ase.Atoms):
        """Convert ASE Atoms to TorchANI format."""
        species = torch.tensor([atoms.get_atomic_numbers()], dtype=torch.long)
        positions = torch.tensor(
            [atoms.get_positions()],
            dtype=torch.float32,
            requires_grad=True
        )
        return species, positions

    if model not in _cached_models:
        _cached_models[model] = getattr(torchani.models, model)(periodic_table_index=True)
        _cached_models[model].eval()

    ani_model = _cached_models[model]
    ani_input = _atoms2ani(atoms)
    energy = ani_model(ani_input).energies

    hess = hessian(
        coordinates = ani_input[1],
        energies = ani_model(ani_input).energies
        )
    return hess



def mace_hessian(atoms:ase.Atoms, model:str='large'):
    """
    Get Hessian using MACE calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure for which the Hessian matrix will be calculated.
    model : str, optional
        The MACE model to use. Options are ['small', 'medium', 'large'].
        Default is 'large'.

    Returns
    -------
    hess : torch.Tensor
        The Hessian matrix as a torch tensor.
    """
    if model not in _cached_models:
        _cached_models[model] = mace_off(
            model=model,
            device='cpu',
            default_dtype='float64'
        )

    mace_model = _cached_models[model]

    mace_hess = mace_model.get_hessian(atoms=atoms)

    return torch.tensor(mace_hess).reshape(1, len(atoms) * 3, -1)


def aimnet2_hessian(atoms:ase.Atoms, model:str='aimnet2', charge=0, mult=1):
    """
    Get Hessian using AIMNet2 calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure for which the Hessian matrix will be calculated.
    model : str, optional
        The AIMNet2 model to use. Options are ['aimnet2'].
        Default is 'aimnet2'.
    charge : int, optional
        The charge of the system. Default is 0.
    mult : int, optional
        The multiplicity of the system. Default is 1.

    Returns
    -------
    hess : torch.Tensor
        The Hessian matrix as a torch tensor.
    """
    if model not in _cached_models:
        _cached_models[model] = AIMNet2Calculator(model=model)

    aimnet2_model = _cached_models[model]

    aimnet2_hess = aimnet2_model(
        data= {
            'coord':torch.tensor(atoms.get_positions()),
            'numbers':torch.tensor(atoms.get_atomic_numbers()),
            'charge':torch.tensor([charge]),
            'mult':torch.tensor([mult])
            },
        forces=True,
        hessian=True
    )['hessian']

    return aimnet2_hess.reshape(1, len(atoms) * 3, -1)


def pyscf_hessian(atoms:ase.Atoms, **calculator_kwargs):
    """
    Get Hessian using PySCF DFT calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure for which the Hessian matrix will be calculated.
    **calculator_kwargs
        Additional keyword arguments to pass to the PySCF calculator.
        - charge (int): The charge of the system.
        - spin (int): The spin of the system.
        - basis (str): The basis set to use.
        - xc (str): The exchange-correlation functional to use.
        - etc

    Returns
    -------
    hess : torch.Tensor
        The Hessian matrix as a torch tensor.
    """

    calc = PySCFCalculator(**calculator_kwargs)
    calc.parameters.verbose = 0
    calc.atoms = atoms

    hess = calc.get_hessian().transpose(0, 2, 1, 3).reshape(1, len(atoms) * 3, -1)

    return torch.tensor(hess)


def atoms2pointgroup(atoms:ase.Atoms, engine='both'):
    """get point group from ase.Atoms
    """
    if engine == 'both':
        pymsym_pg = pymsym.get_point_group(
            positions=atoms.get_positions(),
            atomic_numbers=atoms.get_atomic_numbers()
            )
        pointgroup_pg = PointGroup(
            positions=atoms.get_positions(),
            symbols=atoms.get_chemical_symbols(),
            tolerance_eig=0.01,
            tolerance_ang=4
            ).get_point_group()

        if pymsym_pg == pointgroup_pg:
            return pymsym_pg
        else:
            raise Exception("two pg is not same")
    elif engine == 'pymsym':
        return pymsym.get_point_group(
            positions=atoms.get_positions(),
            atomic_numbers=atoms.get_atomic_numbers()
            )
    elif engine == 'pointgroup':
        return PointGroup(
            positions=atoms.get_positions(),
            symbols=atoms.get_chemical_symbols(),
            tolerance_eig=0.01,
            tolerance_ang=4
            ).get_point_group()
    else:
        raise Exception("engine is not supported")


def is_linear(pointgroup:str)->bool:
    """check if pointgroup is linear
    """
    linear_group = ['Cinfv', 'Dinfh']
    if pointgroup in linear_group:
        return True
    else:
        return False

def vibration_filter(atoms:ase.Atoms, VibAnalysis):
    """
    Filters translational & rotational modes based on the eigenvalues
    Get only vibrationl modes from analysis results.
    """
    pg = atoms2pointgroup(atoms)

    linear_molecule = is_linear(pg)

    # 3N - 5(6), num of vibrations
    degree_of_freedom = len(atoms) * 3
    if linear_molecule:
        N_vibs = degree_of_freedom - 5
    else:
        N_vibs = degree_of_freedom - 6

    freqs = torch.nan_to_num(VibAnalysis.freqs, 0)
    modes = VibAnalysis.modes

    # filter translational & rotational modes
    vib_idx = torch.argsort(freqs, descending=True)[:N_vibs]
    vib_modes = modes[vib_idx]
    vib_freqs = freqs[vib_idx]

    return vib_freqs, vib_modes


def atoms2normal_modes(
    atoms:ase.Atoms,
    engine='torchani',
    mode_type:str='MWN',
    **engine_kwargs
    ):
    """
    Normal modes analyzer

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure for which the Hessian matrix will be calculated.
    engine : str
        The engine to use. Options are ['torchani', 'mace', 'aimnet2', 'pyscf'].
    mode_type : str
        Type of normal mode ( mass weigth & normalization )
        Options are ['MWN', 'MDU', 'MDN']

    """
    if engine == 'torchani':
        hessian = ani_hessian(atoms=atoms, **engine_kwargs)
    elif engine == 'mace':
        hessian = mace_hessian(atoms=atoms, **engine_kwargs)
    elif engine == 'aimnet2':
        hessian = aimnet2_hessian(atoms=atoms, **engine_kwargs)
    elif engine == 'pyscf':
        hessian = pyscf_hessian(atoms=atoms, **engine_kwargs)
    else:
        raise ValueError(f"{engine} is not supported")

    masses = get_atomic_masses(torch.tensor(atoms.get_atomic_numbers()))

    VibAnalysis = vibrational_analysis(
        masses=masses.unsqueeze(0),
        hessian=hessian,
        mode_type=mode_type
    )

    return VibAnalysis


def mode_perturbator(xyz_path:str, normal_mode_vec, amplitude=0.5, output_name=None)->str:
    """
    normal mode like perturbator

    Parameters
    ----------
    - xyz_path : str
        path to xyz file
    - normal_mode_vec : torch.Tensor
        normal mode vector
    - amplitude : float
        amplitude of perturbation
    """
    assert normal_mode_vec.shape[1] == 3, "normal_mode_vec should be (N, 3)"

    _atoms = read(xyz_path, format='xyz')

    coordi = _atoms.get_positions()

    _atoms.set_positions(torch.tensor(coordi) + normal_mode_vec * amplitude)
    xyz_path_name = os.path.basename(xyz_path).split('.')[0]
    xyz_path_name = f"{xyz_path_name + '_perturbed.xyz'}" if output_name is None else output_name

    write(xyz_path_name, _atoms, format='xyz')
