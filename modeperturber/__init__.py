__version__ = '0.0.1'

from .normal_mode_analysis import vibration_filter, atoms2normal_modes, mode_perturbator

#############    setup    #############

import torch
import numpy as np 
import warnings 

torch.set_printoptions(precision=6, sci_mode=False)
np.set_printoptions(precision=6, suppress=True)
warnings.filterwarnings('ignore')

#######################################
