# packages
from typing import (
    Protocol,
    list
)

import numpy as np

import clearwater_modules
from clearwater_modules.tsm.model import EnergyBudget

# clearwater_modules = {
#     'tsm': clearwater_modules.tsm.model,
#     'nsm': clearwater_modules.nsm.model,
# }

class ModuleWrapper(Protocol):
    def __init__(self, **kwargs):
        ...

    def run_step(self, *args) -> np.ndarray:
        ... 


class TSM_Wrapper(ModuleWrapper):
    def __init__(self, **kwargs):
        #instantiate the clearwater module
        self.instance = EnergyBudget(**kwargs)
    
    def run_step(self, *args):
        # modify what we have to work with what is in modules
        self.instance.run_n_timestep(*args)
        return # what you need



# class ModuleWrapper(ABC):
#     def __init__(self, module:clearwater_modules.base.Base, **kwargs):
#         # instatiate the clearwater module 
#         self.instance = module(**kwargs)

#     def run_step(self, *args):
#         # will work what we have
#         self.instance.run_step(*args)
#         return # what you need 
    

# tsm = ModuleWrapper(clearwater_modules.tsm.model, **kwargs)
# nsm = ModuleWrapper(clearwater_modules.nsm.model, **kwargs)

# modules_to_run: list[ModuleWrapper] = [
#     tsm,
#     nsm,
# ]


# provide your boundary conditions, including static things that don't change
# and things that can change timestep to timestep
# each module has dynamic, static, and state variables
# static you define when you instantiate the module and never change
# dynamic variables are calcuated each time step (based on the state variable) --> don't touch 
# state variables; on the modules side 


### def run_timesteps(list_of_modules: list[ModuleWrapper])
# describe int he doc stream that these should be instnatiated module wrappers
# # have instantiated classes in a list
# pass in raw modules from clearwater modules; need to be instantiated     