from pyzebra.anatric import *
from pyzebra.ccl_io import *
from pyzebra.h5 import *
from pyzebra.xtal import *
from pyzebra.ccl_process import *

ZEBRA_PROPOSALS_PATHS = [
    f"/afs/psi.ch/project/sinqdata/{year}/zebra/" for year in (2016, 2017, 2018, 2020, 2021)
]

__version__ = "0.5.0"
