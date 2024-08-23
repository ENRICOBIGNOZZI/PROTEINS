# Protein - Library for Simulating static and dynamic of protein

# Protein is a Python libraryfor Simulating static and dynamic of protein.

## Installation
To install it in `dev` environnement:
```bash
pip install -e <path_to_STATIC>
```
option `-e` it's helpfull to install an editable version. To install in `prod`:
```bash
pip install <path_to_STATIC>
```

## Example of Usage Package for downaloding data from Protein Databank

```python
from Downlaod_data import PDBProcessor
name_of_protein="2m10"
pdb_processor = PDBProcessor(name_of_protein) 
pdb_processor.download_pdb()
pdb_processor.load_structure()
df = pdb_processor.extract_atom_data()
print(df)
```
## exemple of output:
    Residue ID          X          Y          Z  B-Factor  Model ID
0            1 -13.238699  -2.230200 -13.072400   33.6995         0
1            2 -10.020250  -3.929400 -12.479899   37.0770         0
2            3 -10.055600  -7.570600 -12.013100   36.0820         0
3            4  -8.739800  -9.402349  -9.057950   36.5790         0
4            5  -5.009550  -9.626349  -9.308800   43.7120         0
..         ...        ...        ...        ...       ...       ...
## `PDBProcessor` parameters
- `pdb_id`: name of protein to download.



## Example of Usage Package for choosing wich is the best way to legate amino acids

```python
from Visualize import Visualize
visualizer = Visualize(df)
visualizer.plot_connections_vs_radius()
visualizer.calculate_and_print_average_distance()
```
![Number of connections vs radius](docs/img/plot_connections_vs_radius.png)

```python
from Visualize import Visualize
visualizer = Visualize(df)
visualizer.create_and_print_graph(truncated=True, radius=10,plot=True) 
G=visualizer.create_and_print_graph(truncated=False, radius=None,plot=False) 
```



