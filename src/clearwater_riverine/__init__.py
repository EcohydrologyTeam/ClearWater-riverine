# package version
__version__ = '0.6.0'

# populate package namespace
from clearwater_riverine import variables
from clearwater_riverine.io import hdf, inputs, outputs
from clearwater_riverine import mesh, utilities, linalg
from clearwater_riverine.transport import *
