from typing import Callable, Iterator, Union, Optional, List
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging

'''
Set path to the TSM source directory
'''

# Clearwater repo path
repo_path = os.path.dirname(
    os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)))))
print(f'ClearWater repo path: {repo_path}')  # ../../

# TSM path
tsm_path = os.path.join(repo_path, 'modules', 'python', 'src', 'TSM')
print(f'TSM source path: {tsm_path}')  # ../../modules/python/src/TSM
sys.path.append(tsm_path)

import TSM

class Riverine:

    def __init__(self, modules: dict, npts: int, nsteps: int):
        # Initialize debugging log file
        LOG_LEVEL = logging.DEBUG
        LOG_FORMAT = '%(levelname)-8s %(asctime)20s: %(message)s'
        src_path = os.path.dirname(os.path.realpath(__file__))
        log_filename = os.path.join(src_path, 'riverine.log')
        logging.basicConfig(filename=log_filename, level=LOG_LEVEL, format=LOG_FORMAT, filemode='w')
        self.logger = logging.getLogger()

        self.logger.debug(f'__init__(self, {modules}, {npts}, {nsteps})')

        self.modules = [m.lower() for m in modules]
        self.npts = npts
        self.nsteps = nsteps

        '''
        Load and parse the input variables JSON file
        '''
        # Open input parameters file (JSON format)
        # Get the input variables and settings
        self.inputs = {}
        self.outputs = {}
        self.pathways = {}
        for module in self.modules:
            inputs_path = os.path.join(src_path, f'{module}_inputs.json')
            outputs_path = os.path.join(src_path, f'{module}_outputs.json')
            with open(inputs_path) as f:
                self.inputs[module] = json.load(f)
            with open(outputs_path) as f:
                self.outputs[module] = json.load(f)
            self.pathways[module] = {}

    def call_tsm(self, i: int) -> dict:
        '''
        Call the default energy_budget_method() in TSM, passing the values 
        in the input arrays at the current index.

        Parameters:
            i (int) : Current index
        '''

        self.logger.debug(f'call_tsm(self, {i})')

        tsm = TSM.TSM(
            self.inputs['tsm']['TwaterC']['data'][i],
            self.inputs['tsm']['surface_area']['data'][i],
            self.inputs['tsm']['volume']['data'][i],
        )

        tsm.energy_budget_method(
            self.inputs['tsm']['TairC']['data'][i],
            self.inputs['tsm']['q_solar']['data'][i],
            self.inputs['tsm']['pressure_mb']['data'][i],
            self.inputs['tsm']['eair_mb']['data'][i],
            self.inputs['tsm']['cloudiness']['data'][i],
            self.inputs['tsm']['wind_speed']['data'][i],
            self.inputs['tsm']['wind_a']['data'][i],
            self.inputs['tsm']['wind_b']['data'][i],
            self.inputs['tsm']['wind_c']['data'][i],
            self.inputs['tsm']['wind_kh_kw']['data'][i],
            use_SedTemp=self.inputs['tsm']['use_SedTemp']['default'],
            TsedC=self.inputs['tsm']['TsedC']['data'][i],
            num_iterations=10,
            tolerance=0.01
        )

        self.pathways['tsm'][i] = tsm.pathways

    def fake_advection_diffusion_engine(self, variables: dict, mu: float = 0, sigma: float = 0.0001) -> dict:
        '''
        Fake advection-diffusion engine. This applies Gaussian noise to each array.

        Parameters:
            variables (dict[str, str]): Dictionary of variables (could be input or output time series variables)
            mu (float): Mean of Gaussian distribution of noise to apply
            sigma (float): Standard deviation of Gaussian distribution of noise to apply

        Returns:
            Dictionary of advected and diffused input variables

        '''

        self.logger.debug(f'fake_advection_diffusion_engine(self, {variables}, {mu:.2f}, {sigma:.6f})')

        # for variable in variables.keys():
        for variable_name, variable in variables.items():
            # Only operate on floating point variables
            # if variables[variable]['type'] == 'float':
            if variable['type'] == 'float':
                # data = variables[variable]['data']
                data = variable['data']
                # # Add Gaussian random noise
                # noise = np.random.normal(mu, sigma, len(data))
                # data += noise

                # Apply random walk
                dims = 1
                step_n = len(data)
                # step_set = [-0.1, 0.0, 0.1]
                step_set = np.random.normal(mu, sigma, 1000)
                origin = np.zeros((1, dims))
                # Simulate steps in 1D
                step_shape = (step_n, dims)
                steps = np.random.choice(a=step_set, size=step_shape)
                path = np.concatenate([origin, steps]).cumsum(0)

                # variables[variable]['data'] = path
                variable['data'] = path

        return variables

    def compute_water_quality(self):

        self.logger.debug(f'compute_water_quality(self)')

        for module in self.modules:
            # Initialize arrays within the existing input variables dictionary to their specified default values
            for variable in self.inputs[module].keys():
                # Only create arrays for floating point variables
                if self.inputs[module][variable]['type'] == 'float':
                    default = self.inputs[module][variable]['default']
                    self.inputs[module][variable]['data'] = np.full(self.npts, default)

            # Initialize the output variable arrays to zero
            # TODO: change this so that the inputs, main loop, and outputs will operate at different time steps
            # Note: this assumes that all output variables are floating point variables
            for variable in self.outputs[module].keys():
                self.outputs[module][variable]['data'] = np.full(self.npts, 0.0)

            # --- Main Loop ---

            # Iterate through time
            for t in range(self.nsteps):

                # Call kinetics modules for each "cell"
                for i in range(self.npts):
                    # Hard-code this to just call the temperature_and_energy_budget module for now.
                    # TODO: expand this to use the master dictionary mentioned above.
                    if module.lower() == 'tsm':
                        self.call_tsm(i)
                    elif module.lower() == 'gsm':
                        raise NotImplementedError()
                    elif module.lower() == 'nsm1':
                        raise NotImplementedError()
                    elif module.lower() == 'nsm2':
                        raise NotImplementedError()
                    elif module.lower() == 'csm':
                        raise NotImplementedError()
                    elif module.lower() == 'msm':
                        raise NotImplementedError()

                # Compute advection-diffusion
                # TODO: replace with actual advection-diffusion engine, PDE solver, flow and geometry inputs, etc.
                self.fake_advection_diffusion_engine(self.outputs[module])

                # Plot the results
                for variable in self.outputs[module].keys():
                    if self.outputs[module][variable]['type'] == 'float':
                        data = self.outputs[module][variable]['data']
                        plt.figure()
                        plt.plot(data)
                        plt.xlabel('Time Step')
                        units = self.outputs[module][variable]['units']
                        if units == '':
                            units = 'unitless'
                        ystr = '%s (%s)' % (self.outputs[module][variable]['name'], units)
                        plt.ylabel(ystr)
                        outfile = 'test_plot_%s.png' % variable
                        plt.savefig(outfile)
                        plt.title('Time = %d' % t)
