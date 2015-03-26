################################################################
### generic image processing utilities
################################################################

from numpy import *

def norm_mask(v):
    return v/amax(v)
