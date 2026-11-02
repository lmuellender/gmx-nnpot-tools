import torch
from torchani.models import ANI1x, ANI2x
from typing import Optional

try:
    from NNPOps import OptimizedTorchANI
except ImportError:
    OptimizedTorchANI = None

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
            if use_opt == "nnpops":
                assert OptimizedTorchANI is not None, "NNPOps is not available"
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
            cell = cell * self.length_conversion

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
            if use_opt == "nnpops":
                assert OptimizedTorchANI is not None, "NNPOps is not available"
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
            cell = cell * self.length_conversion

        # Run ANI-2x
        result = self.model((atomic_numbers, positions), cell, pbc)

        # Get the potential energy
        energy = result.energies[0] * self.energy_conversion

        # example: compute forces
        grad = torch.autograd.grad([energy], [positions], allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(positions)
        return energy, -1. * grad.squeeze(0)


class GmxANIPairsModel(torch.nn.Module):
    """ANI model that consumes the GROMACS-supplied pairlist directly.

    Unlike the other classes in this module, this one does *not* let TorchANI
    build its own neighborlist. It feeds the externally computed pairs/shifts
    into ``model.compute_from_external_neighbors``, mirroring the interface of
    ``GmxMACEModel``.

    GROMACS supplies a *half* pairlist (each pair appears once, no mirrored
    duplicate), so no duplication is needed. The shift convention matches
    TorchANI's ``narrow_down`` directly: for a pair ``(i, j)`` TorchANI builds
    ``diff = r_i - r_j + shift``, which equals the GROMACS shift unchanged (only
    unit-converted). ``compute_from_external_neighbors`` recomputes the
    distances/diff-vectors from the shifts and re-screens against the model
    cutoff (harmless if the list is already screened; also drops any dummy
    atoms), so ``cell``/``pbc`` are not needed for the energy itself.
    """

    def __init__(self, version=2, aev="pyaev", mnp=False, model_index=None, device=None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        MODEL = ANI1x if version == 1 else ANI2x

        self.model = MODEL(
            neighborlist="all_pairs",  # unused: neighbors are supplied externally
            strategy=aev,
            device=device,
            periodic_table_index=True,
            model_index=model_index,
        ).to(device)
        if mnp:
            self.model.neural_networks.to_infer_model(use_mnp=mnp)
        self.model.requires_grad_(False)

        self.length_conversion = 10.0       # nm (gmx) --> Å (torchani)
        self.energy_conversion = 2625.5     # Hartree (torchani) --> kJ/mol (gmx)

    def forward(self, positions, atomic_numbers, pairs, shifts,
                cell: Optional[torch.Tensor] = None, pbc: Optional[torch.Tensor] = None):

        # Units: nm (gmx) --> Å (torchani)
        positions = positions * self.length_conversion
        shifts = shifts * self.length_conversion

        # TorchANI expects a leading batch ("molecule") dimension
        species = atomic_numbers.unsqueeze(0)            # (1, n_atoms)
        coords = positions.unsqueeze(0)                  # (1, n_atoms, 3)

        # GROMACS half pairlist (n_pairs, 2) --> TorchANI (2, n_pairs).
        # Shifts are passed through unchanged (same sign convention).
        neighbor_idxs = pairs.t().to(torch.long)

        result = self.model.compute_from_external_neighbors(
            species,
            coords,
            neighbor_idxs,
            shifts,
            charge=0,
            atomic=False,
            ensemble_values=False,
        )

        return result.energies[0] * self.energy_conversion


class GmxANIv2Model(torch.nn.Module):
    
    def __init__(self, version=2, aev="cuaev", mnp=False, model_index=None, device=None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        MODEL = ANI1x if version == 1 else ANI2x

        self.model = MODEL(
            neighborlist="adaptive",
            strategy=aev,
            device=device,
            periodic_table_index=True,
            model_index=model_index
        ).to(device)    
        if mnp:
            self.model.neural_networks.to_infer_model(use_mnp=mnp)
        
        self.length_conversion = 10.0       # nm (gmx) --> Å (torchani)
        self.energy_conversion = 2625.5     # Hartree (torchani) --> kJ/mol (gmx)
    
    def forward(self, positions, atomic_numbers, 
                cell: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):

        # Prepare the positions
        species = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if cell is not None:
            cell = cell * self.length_conversion

        # Run ANI-2x
        result = self.model(
            species_coordinates=(species, positions),
            cell=cell,
            pbc=pbc,
            charge=0,
            atomic=False,
            ensemble_values=False,
        )

        return result.energies[0] * self.energy_conversion
