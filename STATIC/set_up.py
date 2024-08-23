from setuptools import setup
import os
current_dir=os.path.dirname(__file__)
before_dir=os.path.abspath(os.path.join(current_dir,os.pardir))
pathDB=os.path.join(before_dir,'DB')
pathCICD=os.path.join(before_dir,'CICD_UTILS')
setup(
    install_requires=[
        "pandas",
        ""
    ]
)
