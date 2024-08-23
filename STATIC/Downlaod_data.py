import requests
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

class PDBProcessor:
    def __init__(self, pdb_id):
        self.pdb_id = pdb_id
        self.file_pdb = f"{pdb_id}.pdb"
        self.url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        self.structure = None

    def download_pdb(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            with open(self.file_pdb, 'wb') as file:
                file.write(response.content)
            print(f"File {self.file_pdb} downloaded and saved successfully!")
        else:
            print(f"Error downloading the PDB file: {response.status_code}")

    def load_structure(self):
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("Molecule", self.file_pdb)

    def extract_atom_data(self):
        if self.structure is None:
            raise ValueError("The structure has not been loaded. Call 'load_structure()' first.")

        #conversion_factor = 8 * np.pi**2 / 3
        atoms = []

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        #bfactor = atom.bfactor
                        #msd = conversion_factor / (bfactor + 1e-2)
                        atoms.append({
                            "Atom Name": atom.name,
                            "Residue Name": residue.resname,
                            "Chain ID": chain.id,
                            "Residue ID": residue.id[1],
                            "X": atom.coord[0],
                            "Y": atom.coord[1],
                            "Z": atom.coord[2],
                            "B-Factor": atom.bfactor,
                            #"MSD": msd,
                            "Model ID": model.id
                        })

        df = pd.DataFrame(atoms)
        return df
    
    def ensamble(self,df):
        grouped = df.groupby(['Residue ID'])
        mean_df = grouped[['X', 'Y', 'Z']].mean().reset_index()
        mean_df['B-Factor'] = grouped['B-Factor'].mean().values
        #mean_df['MSD'] = grouped['MSD'].mean().values
        mean_df['Model ID'] = 0  # Indica che questo è il modello medio
        mean_df=mean_df.dropna().reset_index(drop=True)

        return mean_df
    
