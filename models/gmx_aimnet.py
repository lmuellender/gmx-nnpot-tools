import torch
try:
    from aimnet.calculators.model_registry import get_model_path
except ImportError:
    get_model_path = None

class GmxAIMNet2Model(torch.nn.Module):
    def __init__(self, charge=0, mult=1, **kwargs):
        super().__init__()
        assert get_model_path is not None, "AIMNet2 model requires the aimnet package to be installed."
        model_path = get_model_path("aimnet2")
        self.model = torch.jit.load(model_path)
        self.charge = torch.nn.Parameter(torch.tensor([charge], dtype=torch.float32), requires_grad=False)
        self.mult = torch.nn.Parameter(torch.tensor([mult], dtype=torch.int32), requires_grad=False)
        self.length_conversion = 10.0       # nm (gmx) --> Å (aimnet)
        self.energy_conversion = 96.4853    # eV (aimnet) --> kJ/mol (gmx)

    def forward(self, positions, atomic_numbers, cell):
        # Prepare the model input
        positions = positions * self.length_conversion
        if cell is not None:
            cell = cell * self.length_conversion
        else:
            cell = torch.zeros(3, 3).to(positions.device)

        # Prepare input for aimnet model
        input_data = {
            "coord": positions.unsqueeze(0),
            "numbers": atomic_numbers.unsqueeze(0),
            "charge": self.charge,
            "mult": self.mult,
            "cell": cell,
        }
        result = self.model(input_data)

        energy = result['energy'] * self.energy_conversion  # eV --> kJ/mol

        return energy
