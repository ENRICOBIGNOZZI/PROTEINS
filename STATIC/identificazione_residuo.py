from Bio.PDB import PDBParser

# Carica il file PDB
pdb_parser = PDBParser(QUIET=True)
structure = pdb_parser.get_structure("3LNX", "/Users/enrico/PROTEINS/STATIC/3LNX copy.pdb")

# Conta i carboni alpha nella chain A
alpha_carbons = []
for model in structure:
    for chain in model:
        if chain.id == "A":  # Considera solo la catena A
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA":
                        alpha_carbons.append({
                            "residue_name": residue.resname,
                            "residue_number": residue.id[1],
                            "atom_serial": atom.serial_number
                        })

# Totale di carboni alpha
total_alpha_carbons = len(alpha_carbons)

# Residui del sito attivo e allosterici
active_site_residues = [347, 348, 349]  # Residui del sito attivo
allosteric_site_residues = [347, 362, 367, 372]  # Residui allosterici

# Identifica le posizioni relative dei residui nella lista CA
active_site_positions = [
    idx for idx, atom in enumerate(alpha_carbons)
    if atom["residue_number"] in active_site_residues
]

allosteric_site_positions = [
    idx for idx, atom in enumerate(alpha_carbons)
    if atom["residue_number"] in allosteric_site_residues
]

print("Totale CA nella chain A:", total_alpha_carbons)
print("Posizioni sito attivo nella chain A:", active_site_positions)
print("Posizioni siti allosterici nella chain A:", allosteric_site_positions)
