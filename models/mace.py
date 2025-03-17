import torch
from typing import Optional

import mace
from mace import scatter_sum


class GmxMACEModel(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.z_table = model.atomic_numbers.tolist()
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        if not hasattr(model, "heads"):
            model.heads = [None]
        self.register_buffer(
            "head",
            torch.tensor(
                self.model.heads.index(kwargs.get("head", self.model.heads[-1])),
                dtype=torch.long,
            ).unsqueeze(0),
        )

        self.length_conversion = 10.0       # nm (gmx) --> Å (mace)
        self.energy_conversion = 96.4853    # eV (mace) --> kJ/mol (gmx)
    
    def forward(self, positions, atomic_numbers, pairs, shifts,
                cell: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):
        
        # Prepare the model input
        positions = positions * self.length_conversion
        n_atoms = positions.shape[0]
        device = positions.device
        if cell is not None:
            cell = cell * self.length_conversion
        else:
            cell = torch.zeros(3, 3).to(device)

        # transpose and duplicate pairs
        pairs = torch.cat([pairs, pairs[:, [1, 0]]], dim=0).t().to(torch.int64)
        shifts = torch.cat([-shifts, shifts], dim=0) * self.length_conversion

        # One hot encoding of atomic numbers
        # no need to convert since gromacs and mace use the same atomic numbers
        nodeAttrs = torch.zeros(n_atoms, len(self.z_table), device=device)
        indices = torch.stack([torch.tensor(self.z_table.index(z)) for z in atomic_numbers])
        nodeAttrs[torch.arange(n_atoms), indices] = 1.0

        # other inputs
        ptr = torch.tensor([0, n_atoms], dtype=torch.int64, requires_grad=False, device=device)
        batch = torch.zeros(n_atoms, dtype=torch.int64, device=device)
        if pbc is None:
            pbc = torch.tensor([True, True, True], requires_grad=False, device=device)
        
        # Prepare the input dict
        input_data = {
            "ptr": ptr,
            "node_attrs": nodeAttrs,
            "batch": batch,
            "positions": positions,
            "edge_index": pairs,
            "shifts": shifts,
            "pbc": pbc,
            "cell": cell,
        }

        # run the model
        out = self.model(
            input_data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False
        )

        total_energy = out["energy"]
        if total_energy is None:
            total_energy = torch.tensor(0.0, device=device)

        return total_energy
