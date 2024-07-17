import os

__version__ = "0.1.0"

"""
# set Python env variable to keep track of example data dir
pyALF_dir = os.path.dirname(__file__)
DATADIR = os.path.join(pyALF_dir, "Example/")

# try to import many modules

try: 
    from astropy import units as u
    import astropy.constants as c
    from astropy import constants as const
    astropy_ext = True
except:
    astropy_ext = False

try:
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter
    from scipy.signal import argrelextrema
    from numpy import median
    import math
    import matplotlib.pyplot as plt
    import re
    import shutil
    import string
    req_ext = True
except:
    req_ext = False

"""