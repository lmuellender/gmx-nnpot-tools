import torch
from typing import Optional
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdPartialCharges

NUTMEG_MODEL_FILE = "/home/lukas/Documents/nnpot/models/nutmeg-small-raw.pt"

def gasteigerChargesFromGroFile(gro_file, total_charge=0):
    """ Attempts to read a .gro file and compute Gasteiger charges for the atoms in it.
    This is needed to create the input features for the Nutmeg model.
    """
    # Load the .gro file with MDAnalysis
    u = mda.Universe(gro_file)
    u.guess_TopologyAttrs(to_guess=["elements", "bonds"])
    # Create an RDKit molecule to compute Gasteiger charges
    rdmol = Chem.EditableMol(Chem.Mol())
    for atom in u.atoms: # type: ignore
        a = Chem.Atom(atom.element)
        a.SetNoImplicit(True)
        rdmol.AddAtom(a)
    for bond in u.bonds:
        i, j = int(bond.atoms[0].index), int(bond.atoms[1].index)
        rdmol.AddBond(i, j, Chem.BondType.SINGLE)
    rdmol = rdmol.GetMol()
    Chem.SanitizeMol(rdmol)
    rdDetermineBonds.DetermineBondOrders(rdmol, total_charge, embedChiral=False)
    rdPartialCharges.ComputeGasteigerCharges(rdmol)
    charges = [rdmol.GetAtomWithIdx(int(atom.index)).GetDoubleProp('_GasteigerCharge') for atom in u.atoms] # type: ignore
    elements = [atom.element for atom in u.atoms] # type: ignore
    return elements, charges


def create_atom_features(symbols, charges):
    """ ADAPTED FROM https://github.com/openmm/nutmeg/blob/main/nutmegpotentials/util.py
    
    This utility function creates the input tensors that need to be passed
    to a Nutmeg model's forward() method to describe the atoms.
    
    Parameters
    ----------
    symbols: list of str
        the element symbol for each atom in the system to be simulated
    charges: list of float
        the Gasteiger partial charge of each atom in the system to be simulated

    Returns
    -------
    types: torch.Tensor
        the type index of each atom
    node_attrs: torch.Tensor
        the feature vector for each atom
    """
    typeDict = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Si': 9, 'P': 10, 'S': 11, 'Cl': 12, 'K': 13, 'Ca': 14, 'Br': 15, 'I': 16}
    types = torch.tensor([typeDict[symbol] for symbol in symbols], dtype=torch.int64)
    one_hot_z = torch.nn.functional.one_hot(types, num_classes=17).to(torch.float32)
    charges = torch.tensor([[c] for c in charges], dtype=torch.float32)
    node_attrs = torch.cat([one_hot_z, charges], dim=1)
    return types, node_attrs

class GmxNutmegModel(torch.nn.Module):
    def __init__(self, types, node_attrs):
        super().__init__()
        self.model = torch.load(NUTMEG_MODEL_FILE, weights_only=False)
        self.types = torch.nn.Parameter(types, requires_grad=False)
        self.node_attrs = torch.nn.Parameter(node_attrs, requires_grad=False)
        self.length_conversion = 10.0       # nm (gmx) --> Å (nutmeg)
        self.energy_conversion = 96.4853321 # eV (nutmeg) --> kJ/mol (gmx)

    def forward(self, positions, atomic_numbers, cell: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):
        # Prepare the positions
        atomic_numbers = atomic_numbers
        positions = positions * self.length_conversion # nm --> Å
        if cell is not None:
            cell *= self.length_conversion  # nm --> Å

        # Run the model
        # with torch.jit.optimized_execution(False): # pyright: ignore[reportPrivateImportUsage]
        energy = self.model(positions, self.types, self.node_attrs, cell)

        return energy * self.energy_conversion


if __name__ == "__main__":
    # test charge and symbol extraction from gro file
    gro_file = "/home/lukas/Documents/nnpot/test/alanine/ala2.gro"
    # gro_file = "/home/lukas/Documents/nnpot/benchmark/water_new/waters_100/em.gro"
    symbols, charges = gasteigerChargesFromGroFile(gro_file)
    print("Symbols: ", symbols)
    print("Charges: ", charges)

    # test model export
    types, node_attrs = create_atom_features(symbols, charges)
    model = GmxNutmegModel(types, node_attrs)
    model.eval()
    model.to("cpu")
    dummy_positions = torch.randn(len(symbols), 3, dtype=torch.float32, device="cpu", requires_grad=True)
    dummy_atomic_numbers = torch.tensor([1,6,1,1,6,8,7,1,6,1,6,1,1,1,6,8,7,1,6,1,1,1], dtype=torch.int64, device="cpu") # atomic numbers for alanine dipeptide 
    dummy_cell = torch.tensor([[2.9, 0.0, 0.0],
                               [0.0, 2.9, 0.0],
                               [0.0, 0.0, 2.9]], dtype=torch.float32, device="cpu")
    energy = model(dummy_positions, dummy_atomic_numbers, cell=dummy_cell)[0]
    print("Energy: ", energy.item())

    scripted_model = torch.jit.script(model)
    scripted_model.save("models/nutmeg_ala2.pt")
    
