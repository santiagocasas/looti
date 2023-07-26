from looti import datahandle as dhl
from looti import dictlearn as dcl
import numpy as np

from copy import deepcopy

from looti.cosmo_emulator import CosmoEmulator



class LootiFish(CosmoEmulator):


    def __init__(self, training_args=dict()):
        super().__init__()
        self.training_args = training_args
        # Read data from path/csv files for each cosmological quantity in extra_args
        self.must_provide = ['Plin', 'Pnonlin', 'sigma8', 'f_GrowthRate', 'background_H']
        for quantity in self.training_args['quantities']:
            if quantity in self.must_provide:
            # If there exists path to trained intobj, read it
                if 'emulator_path' in self.training_args['quantities'][quantity].keys():
                    self.read_emulator(cosmo_quantity=quantity, 
                                       data_path=self.training_args['quantities'][quantity]['data_path'],
                                       emulator_path=self.training_args['quantities'][quantity]['emulator_path']
                                       )
        
        #self.z_grid = training_args['quantities']['z_grid']
        self.z_grid = 
        return None

    def set_args(self, kwargs):
        self.training_args = kwargs
                
    def get_kgrid(self, params_values_dict, units="Mpc"):
        # Get Pk from intobj
        kgrid, Pk = self.get_prediction(cosmo_quantity='Plin',
                                        input_dict=params_values_dict)
        return kgrid 
    
    def get_zgrid(self, params_values_dict, units="Mpc"):
        # Get Pk from intobj
        zgrid, Hub = self.get_prediction(cosmo_quantity='background_H',
                                        input_dict=params_values_dict)
        return zgrid 
    
    def compute_Pk(self, params_values_dict, nonlinear=False, units="Mpc"):
        # Get Pk from intobj
        if nonlinear == False:
            kgrid, Pk = self.get_prediction(cosmo_quantity='Plin',
                                            input_dict=params_values_dict)
        elif nonlinear ==  True:
            kgrid, Pk = self.get_prediction(cosmo_quantity='Pnonlin',
                                            input_dict=params_values_dict)
        
        Pkz = np.array((len(self.z_grid), len(kgrid)))
        for ii,zz in enumerate(self.z_grid):
            Pkz[ii, :] = Pk
        
        return Pkz
                

