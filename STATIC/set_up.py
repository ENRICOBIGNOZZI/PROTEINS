from setuptools import setup
import os
current_dir=os.path.dirname(__file__)
before_dir=os.path.abspath(os.path.join(current_dir,os.pardir))
setup(
    install_requires=[
        "biopython",
        "pandas",
        "numpy",
        "requests",
        "networkx",
        "matplotlib.pyplot",
        "seaborn",
        "scipy",
        "os",


    ]
)
