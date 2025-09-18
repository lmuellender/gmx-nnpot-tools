import torch
import numpy as np
import argparse as ap
import os
try:
    from models.gmx_ani import GmxANIModel
except ImportError:
    GmxANIModel = None
try:
    from models.gmx_mace import GmxMACEModel, GmxMACEModelNoPairs
except ImportError:
    GmxMACEModel, GmxMACEModelNoPairs = None, None
try:
    from models.gmx_aimnet import GmxAIMNet2Model
except ImportError:
    GmxAIMNet2Model = None
    GmxAIMNet2Model = None
try:
    from models.gmx_emle import GmxEMLEModel
except ImportError:
    GmxEMLEModel = None
try:
    from models.gmx_nutmeg import GmxNutmegModel, gasteigerChargesFromGroFile, create_atom_features
except ImportError:
    GmxNutmegModel, gasteigerChargesFromGroFile, create_atom_features = None, None, None

# set this value to true to export a mace model that does not need pair input
_use_nnpops_pairs = False

_map_atom_number = {
    "h": 1,
    "c": 6,
    "n": 7,
    "o": 8,
    "f": 9,
    "s": 16,
    "cl": 17,
}

def atomNumberFromLine(line):
    atype = line.split()[1].lower()
    # Sort keys by length descending to match 'cl' before 'c'
    for key in sorted(_map_atom_number.keys(), key=len, reverse=True):
        if atype.startswith(key):
            return _map_atom_number[key]
    raise ValueError(f"Atom number couldn't be inferred from line: {line}")

def getIndices(ndxfile, group=None):
    """
    Get the atom indices from a GROMACS index file for a given group.
    If group is None, use the 'non-Water' group.
    """

    # Read the GROMACS index file
    with open(ndxfile, 'r') as f:
        lines = f.readlines()

    # Find the group of interest
    if group is None:
        group = "non-Water"
        print("Group not specified, using 'non-Water' group.")
    else:
        print(f"Reading group {group} from index file..")

    group_lines = [line for line in lines if line.startswith(f"[ {group} ]")]
    if not group_lines:
        raise ValueError(f"Group '{group}' not found in index file.")
    
    start_index = lines.index(group_lines[0]) + 1
    # Find the end of the group, i.e. next group or end of file
    end_index = start_index
    while end_index < len(lines) and not lines[end_index].startswith("["):
        end_index += 1
    indices = [int(num) for line in lines[start_index:end_index] for num in line.split()]

    return indices

def getAtomNumbers(grofile, ndxfile=None, group=None):
    """
    Get the atomic numbers from the GROMACS index and coordinate files.
    """

    indices = []
    if ndxfile is not None:
        indices = getIndices(ndxfile, group)
    if group is None:
        group = ""

    # Read the GROMACS coordinate file
    with open(grofile, 'r') as f:
        lines = f.readlines()[2:]  # Skip the first two lines (title and number of atoms)

    # Get atomic numbers based on indices
    atomic_numbers = []
    if indices:
        for idx in indices:
            for line in lines:
                if int(line.split()[2]) == idx:
                    atomic_numbers.append(atomNumberFromLine(line))
                    break
    elif group.lower() == "system":
        # If the group is "System", use all atoms
        for line in lines[:-1]:  # Skip the last line (box dimensions)
            atomic_numbers.append(atomNumberFromLine(line))
    else:
        # If not, use all atoms until the first water or EOF
        for line in lines:
            if "SOL" in line or len(line.split()) <= 3:
                break
            # get the atom type and infer number, assuming the first part is the atom type
            atomic_numbers.append(atomNumberFromLine(line))
    assert atomic_numbers, "Atom numbers could not be read from the coordinate file."
    print(f"Read atom numbers from coordinate file: {' '.join(map(str, atomic_numbers))}")
    return torch.tensor(atomic_numbers, dtype=torch.int64)

def checkExtensions():
    """
    Check for loaded torch extension libraries and return a dict of them.
    """
    ext_lib = []
    for lib in torch.ops.loaded_libraries:
        if lib:
            ext_lib.append(lib)
    ext_lib = ":".join(ext_lib)
    print("Loaded extension libraries: ", ext_lib)
    extra_files = {}
    if ext_lib:
        extra_files['extension_libs'] = ext_lib
    return extra_files

def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the atomic numbers from the gromacs index and coordinate files
    if args.grofile is not None and "nutmeg" not in args.model:
        atomic_numbers = getAtomNumbers(args.grofile, args.ndxfile, args.group)
    else:
        atomic_numbers = None

    # Create the model
    if "ani" in args.model:
        assert GmxANIModel is not None, "ANI model is not available. Please install the required dependencies."
        if "emle" in args.model:
            assert GmxEMLEModel is not None, "EMLE wrapper for ANI is not available. Please install the required dependencies."
            model = GmxEMLEModel(flavor="ani2x", atomic_numbers=atomic_numbers, model_index=args.model_index, dtype=torch.float32, device=device)
            print(f"Saving ANI2x-EMLE model to {args.outfile}")
        elif args.model == "ani1x":
            model = GmxANIModel(use_opt=args.use_opt, atomic_numbers=atomic_numbers, model_index=args.model_index, version=1, device=device)
            print(f"Saving ANI-1x model with {args.use_opt} optimization to {args.outfile}")
        elif args.model == "ani2x":
            model = GmxANIModel(use_opt=args.use_opt, atomic_numbers=atomic_numbers, model_index=args.model_index, version=2, device=device)
            print(f"Saving ANI-2x model with {args.use_opt} optimization to {args.outfile}")
        else:
            raise ValueError("Invalid model name for ANI: {}".format(args.model))

    elif "mace" in args.model:
        assert GmxMACEModel is not None, "MACE model is not available. Please install the required dependencies."
        if "emle" in args.model:
            assert GmxEMLEModel is not None, "EMLE wrapper for MACE is not available. Please install the required dependencies."
            model = GmxEMLEModel(flavor="mace", atomic_numbers=atomic_numbers, model_index=args.model_index, dtype=torch.float32, device=device)
            print(f"Saving MACE-EMLE model with {args.use_opt} optimization to {args.outfile}")
        else:
            if GmxMACEModelNoPairs is not None:
                model = GmxMACEModelNoPairs(size="small", device=device)
            else:
                model = GmxMACEModel(size="small", device=device)
            print(f"Saving MACE model to {args.outfile}")

    elif "aimnet" in args.model:
        assert GmxAIMNet2Model is not None, "AIMNet2 model is not available. Please install the required dependencies."
        model = GmxAIMNet2Model(charge=0, mult=1).to(device)
        print(f"Saving AIMNet2 model to {args.outfile}")

    elif "nutmeg" in args.model:
        assert GmxNutmegModel is not None, "Nutmeg model is not available. Please install the required dependencies."
        assert args.grofile is not None, "A .gro file must be provided to compute Gasteiger charges for the Nutmeg model."
        symbols, charges = gasteigerChargesFromGroFile(args.grofile, total_charge=0) # pyright: ignore[reportOptionalCall]
        types, node_attrs = create_atom_features(symbols, charges) # pyright: ignore[reportOptionalCall]
        model = GmxNutmegModel(types, node_attrs)
        print(f"Saving Nutmeg model to {args.outfile}")

    else:
        raise ValueError("Invalid model name: {}".format(args.model))
    
    # Check for extensions
    extensions = checkExtensions()
    
    # Save the model
    assert args.outfile.endswith(".pt"), "Output file name must end with .pt"
    torch.jit.script(model).save(args.outfile, _extra_files=extensions)
    print(f"Saved wrapped model to {args.outfile}.")


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Wrap a NNP model for use in GROMACS.")
    parser.add_argument("model", type=str, help="Model to wrap. Should be one of ani1x, ani2x, ani2x_emle, mace, mace_emle, aimnet, nutmeg.")
    parser.add_argument("--outfile", type=str, default="model_gmx.pt", help="Output file name")
    parser.add_argument("--use-opt", type=str, default=None, help="Use optimization (cuaev, nnpops)")
    parser.add_argument("--group", type=str, default=None, help="Index group name")
    parser.add_argument("--ndxfile", type=str, default=None, help="GROMACS Index file")
    parser.add_argument("--grofile", type=str, default=None, help="GROMACS Coordinate file")
    parser.add_argument("--model-index", type=int, default=None, help="Model index")

    # Parse the arguments
    args = parser.parse_args()

    main(args)
