from looti import datahandle as dhl
from looti import dictlearn as dcl
import numpy as np

from copy import deepcopy

from looti.cosmo_emulator import CosmoEmulator



class LootiFish(CosmoEmulator):


    def __init__(self, training_args=dict()):
        self.training_args = training_args
        external_info = training_args['external_info']
        super().__init__(external_info=external_info)
        # Read data from path/csv files for each cosmological quantity in extra_args
        self.must_provide = ['Plin', 'Pnonlin', 'sigma8', 
                             'f_GrowthRate', 'D_Growth', 
                             'background_H']
        self.directory = self.training_args['directory']
        for quantity in self.training_args['quantities']:
            if quantity in self.must_provide:
            # If there exists path to trained intobj, read it
                self.read_emulator(cosmo_quantity=quantity, 
                                   directory=self.directory
                                  )
        self.cosmo_params = self.training_args['parameters']
        self.cosmo_param_basis = self.training_args['parameter_basis'] 
        self.P_z_values = []
        self.z_grid =  self.get_fgrid('background_H')
        self.k_grid =  self.get_fgrid('Plin')

    def set_args(self, kwargs):
        self.training_args = kwargs
                
    def compute_P(self, params_values_dict, nonlinear=False, units="Mpc"):
        # Get Pk from intobj
        if nonlinear == False:
            z_vals, kgrid, Pk = self.get_prediction(cosmo_quantity='Plin',
                                            input_dict=params_values_dict)
        elif nonlinear ==  True:
            z_vals, kgrid, Pk = self.get_prediction(cosmo_quantity='Pnonlin',
                                            input_dict=params_values_dict)
        
        self.P_z_values = z_vals
        #print(len(z_vals))
        #print(len(kgrid))
        Pkz = np.empty((len(z_vals), len(kgrid)))
        #print(Pkz.shape)
        for ii, zz in enumerate(z_vals):
            #print(Pk[ii].shape)
            Pkz[ii, :] = Pk[ii]
        
        return Pkz
    
    def compute_fGrowthRate(self, params_values_dict, units="Mpc"):
        z_vals, kgrid, fk = self.get_prediction(cosmo_quantity='f_GrowthRate',
                                            input_dict=params_values_dict)
        
        self.fGrowthRate_z_values = z_vals
        fkz = np.empty((len(z_vals), len(kgrid)))
        for ii,zz in enumerate(z_vals):
            fkz[ii, :] = fk[ii]
        
        return fkz
    
    def compute_DGrowth(self, params_values_dict, units="Mpc"):
        z_vals, kgrid, Dk = self.get_prediction(cosmo_quantity='D_Growth',
                                            input_dict=params_values_dict)
        
        self.DGrowth_z_values = z_vals
        Dkz = np.empty((len(z_vals), len(kgrid)))
        for ii,zz in enumerate(z_vals):
            Dkz[ii, :] = Dk[ii]
        
        return Dkz
                
    def compute_background_H(self, params_values_dict, units="Mpc"):
        z_vals, zgrid, Hz = self.get_prediction(cosmo_quantity='background_H',
                                            input_dict=params_values_dict)
        ## get prediction returns a list of Hz at z_vals
        # # in this case z_vals=[0] and the list has one element 
        return Hz[0]

    def compute_sigma8(self, params_values_dict, units="Mpc"):
        z_vals, zgrid, s8z = self.get_prediction(cosmo_quantity='sigma8',
                                            input_dict=params_values_dict)
        return s8z[0]
