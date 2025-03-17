import torch
from torchani.models import ANI1x, ANI2x
from typing import Optional

try:
    from NNPOps import OptimizedTorchANI
    _has_nnpops = True
except ImportError:
    _has_nnpops = False

import os


class GmxANIModel(torch.nn.Module):
    
    def __init__(self, use_opt=None, atomic_numbers=None, model_index=None, version=1, device=None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if version == 1:
            self.model = ANI1x(periodic_table_index=True, model_index=model_index).to(device)
        elif version == 2:
            self.model = ANI2x(periodic_table_index=True, model_index=model_index).to(device)
        else:
            raise ValueError("Invalid ANI version number")

        if use_opt is not None:
            if use_opt == "cuaev":
                self.model.aev_computer.use_cuda_extension = True
                self.model.aev_computer.cuaev_enabled = True
                self.model.aev_computer.init_cuaev_computer()
            elif use_opt == "nnpops":
                assert _has_nnpops, "NNPOps is not available"
                assert atomic_numbers is not None, "Atomic numbers must be provided for NNPOps"
                self.model = OptimizedTorchANI(self.model, atomic_numbers.unsqueeze(0)).to(device)
            else:
                raise ValueError("Invalid option for use_opt: Must be 'cuaev', 'nnpops', or None")
            
        self.length_conversion = 10.0       # nm (gmx) --> Å (torchani)
        self.energy_conversion = 2625.5     # Hartree (torchani) --> kJ/mol (gmx)
    
    def forward(self, positions, atomic_numbers, 
                cell: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):

        # Prepare the positions
        atomic_numbers = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if cell is not None:
            cell *= self.length_conversion

        # Run ANI-2x
        result = self.model((atomic_numbers, positions), cell, pbc)
        
        # Get the potential energy
        energy = result.energies[0] * self.energy_conversion
    
        return energy


class GmxANIForceModel(torch.nn.Module):
    
    def __init__(self, use_opt=None, atomic_numbers=None, model_index=None, version=1, device=None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if version == 1:
            self.model = ANI1x(periodic_table_index=True, model_index=model_index).to(device)
        elif version == 2:
            self.model = ANI2x(periodic_table_index=True, model_index=model_index).to(device)
        else:
            raise ValueError("Invalid ANI version number")

        if use_opt is not None:
            if use_opt == "cuaev":
                self.model.aev_computer.use_cuda_extension = True
                self.model.aev_computer.cuaev_enabled = True
                self.model.aev_computer.init_cuaev_computer()
            elif use_opt == "nnpops":
                assert _has_nnpops, "NNPOps is not available"
                assert atomic_numbers is not None, "Atomic numbers must be provided for NNPOps"
                self.model = OptimizedTorchANI(self.model, atomic_numbers.unsqueeze(0)).to(device)
            else:
                raise ValueError("Invalid option for use_opt: Must be 'cuaev', 'nnpops', or None")
            
        self.length_conversion = 10.0       # nm (gmx) --> Å (torchani)
        self.energy_conversion = 2625.5     # Hartree (torchani) --> kJ/mol (gmx)
    
    def forward(self, positions, atomic_numbers, 
                cell: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):

        # Prepare the positions
        atomic_numbers = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if cell is not None:
            cell *= self.length_conversion

        # Run ANI-2x
        result = self.model((atomic_numbers, positions), cell, pbc)
        
        # Get the potential energy
        energy = result.energies[0] * self.energy_conversion
    
        # example: compute forces
        grad = torch.autograd.grad([energy], [positions], allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(positions)
        return energy, -1. * grad.squeeze(0)
