# package version
__version__ = '0.6.0'

# populate package namespace
from . import variables
from .io import hdf, inputs, outputs
from . import mesh, utilities, linalg
from .model import ClearwaterRiverine
