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
![Number of connections vs radius](docs/img/plot_connections_vs_radius.png'.png)

```python
from Visualize import Visualize
visualizer = Visualize(df)
visualizer.create_and_print_graph(truncated=True, radius=10,plot=True) 
G=visualizer.create_and_print_graph(truncated=False, radius=None,plot=False) 
```



# Example of downloading prices verbosly from Interactive Brokers
prices_data = download_data(DownloadedPrices, contract, end_date='2022-01-05', start_date='2022-01-01', verbose=True, isToDownloadFromIB=True)

# Store data persistently on file
prices_data.store()
# To store on DB inser `engine` as parameter


# Example of downloading data from a sample file (for debugging) specifing a relative timespan of 23 days
debug_news = download_data(DownloadedNews, contract, end_date='2022-01-05', off=23, isToDownloadFromIB=False)

# Store data in a database
engine = create_engine("sqlite+pysqlite://localhost/")
prices_data.store(engine)
```
## `download_data` parameters

The `download_data` function accepts the following parameters:
- `typeData`: data type to download (possible are DownloadedPices, DownloadedNews, DownloadedTicks and DownloadedLevels)
- `contract`: Contract with instrument symbol in it
- `end_date`: limit date of timespan of data to download. Defaults to `datetime.now()`.
- `off`: offset for relative timespan back in time respect `end_date`; can be specified as `int` (days) or as `datetime.dimedelta`. Defaults to None.
- `start_date`: start date of timespan of data to download for an absolute timespan determination. Defaults to None.
- `verbose`: log verbosly. Defaults to False.
- `silent`: subpress warnings. Defaults to False.
- `isToDownloadFromIB`: True means that data is downloaded from IB, False from example file (ONLY DEBUG PURPOUSE). Defaults to True.
- `symbol`: Symbol of the financial security of interest.
- `data_type`: Type of data to download (prices, news, ticks, levels).
- `start_date`: Start date for data download in 'YYYY-MM-DD' format.

this function return one instance of a subclass of `DownloadedData`:
- `DownloadedPrices`
- `DownloadedNews`
- `DownloadedTicks`
- `DownloadedLevels`

## `store` method parameters

The `store` method supports various options to customize data storage. It is implemented for all subclasses of `DownloadedData`
-  `engine`: Database engine, if `None` it stores on file. Defaults to None.
- `verbose`: log verbosly. Defaults to False.
- `silent`: subpress warnings. Defaults to False.


## Example of Usage Command-Line Interface
```bash
# Download prices from Interactive Brokers and save to file for all 2022 year (first `end_date`, then `start_date`)
delphydd --online --onfile prices 2022-12-31 2022-01-01

# Download ticks from a sample file (for debugging) and save to file silently (default timespan is from current date, 1 month back)
delphydd -s --offline --onfile ticks 

# Store data in a database (relative timespan of 3 days specified)
delphydd --offset 3 --onDB levels 
```

## Command-Line Options

The `delphydd` command helps download data from Interactive Broker and store it on DB or on file. 

It wants these positional arguments:
- `<data_type>`: type of data to download, can be one between `prices`,`news`,`ticks` or `levels`
- `end_date`: download data until this date, default is current date
- `start_date`: download data from this date. Used to specify an absolute timespan, if not setted, relative timespan is used

It accept these options:
- `-h`, `--help`: show the help message and exit
- `--online`, `-i`: download data from interactive broker
- `--offline`, `-I`: download data from example file
- `--onfile`, `-f`: store data to file
- `--onDB`, `-d`: store data to database
- `--offset <OFFSET>`, `-o <OFFSET>`: relative timedelta from `end_date` in days to specify relative timespan of data to be downloaded, if not setted and absolute timespan in not used, default is 30 days
- `--verbose`, `-v`: log data downloaded and other info
- `--silent`, `-s`: subpress any log, also warnings

The option `--verbose` can not be specified with `--silent`, `--online` can not be specified with `--offline`, `--onDB` with `--onfile` and `--offset` with `start_date` due to conflit between these options.

