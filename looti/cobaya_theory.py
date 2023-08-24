from looti import datahandle as dhl
from looti import dictlearn as dcl
import numpy as np

from copy import deepcopy

from looti.cosmo_emulator import CosmoEmulator
from cobaya.theories.cosmo import BoltzmannBase

from collections import deque
from typing import Sequence, Optional, Union, Tuple, Dict, Iterable, Set, Any, List


class Looti_Cobaya(CosmoEmulator,BoltzmannBase):

    _input_params_extra: Set[str] = set()
    _states: deque


    def __init__(self):
        super().__init__()
        self._input_params_extra = set()
        self._states = deque()
        self._must_provide = set()

        self._name = 'looti'

        self.set_logger('looti')
        self.set_timing_on(True)


    def initialize(self):
        
        print('initialize')
        super().initialize()
        self.set_timing_on(True)

        # Read data from path/csv files for each cosmological quantity in extra_args


        for quantity in self.extra_args['quantities']:

            # If there exists path to trained intobj, read it

            if 'emuobj_directory' in self.extra_args['quantities'][quantity].keys():

                print('LOOTI: Loading emulator for ', quantity)
                self.read_emulator(cosmo_quantity=quantity, directory=self.extra_args['quantities'][quantity]['emuobj_directory'])
                print('LOOTI: Loaded')


            # Else, read data from csv files and train intobj
            else:
                print('LOOTI: Loading data from csv files')
                self.read_data(quantity, 
                            data_path = self.extra_args['quantities'][quantity]['data_path'],
                                file_name = self.extra_args['quantities'][quantity]['file_name'],
                                n_params = self.extra_args['quantities'][quantity]['n_params'],
                                n_train = self.extra_args['quantities'][quantity]['n_train'],
                                n_test = self.extra_args['quantities'][quantity]['n_test'],
                                features_to_Log = self.extra_args['quantities'][quantity].get('features_to_Log', True),
                                observable_to_Log = self.extra_args['quantities'][quantity].get('observable_to_Log', False),
                                **self.extra_args['quantities'][quantity].get('kwargs', {}))
                print('LOOTI: Data loaded')
                print('LOOTI: Training emulator for ', quantity)
                # Train with create intobj
                self.create_emulator(quantity, 
                                n_params = self.extra_args['quantities'][quantity]['n_params'], 
                                **self.extra_args['quantities'][quantity].get('kwargs', {}))
                print('LOOTI: Training complete.')
            


    def set_args(self, kwargs):
        self.extra_args = kwargs

    def must_provide(self, **requirements):
        # Computed quantities required by the likelihood

        #TODO: USE ONCE get_info is implemented

        for req, v in requirements.items():

            if not req in self._must_provide.keys():
                    self._must_provide[req] = {}

            if req == 'Cl':
                if 'tt' in v.keys():
                    if not 'TT' in self.emu_objs.keys():
                        raise ValueError("You must provide the Cl's for TT")
                    #elif v['tt']>self.get_info('ttcl')['grid_max']:
                    #    raise ValueError("Required Cl's (%d) for TT are larger than the ones in the emulator (%d)", v['tt'], self.get_info('ttcl')['grid_max'])
                    if 'tt' in self._must_provide['Cl'].keys():
                        if self._must_provide['Cl']['tt'] < v['tt']:
                            self._must_provide['Cl']['tt'] = v['tt']
                    else:
                        self._must_provide['Cl']['tt'] = v['tt'] 
                if 'te' in v.keys():
                    if not 'TE' in self.emu_objs.keys():
                        raise ValueError("You must provide the Cl's for TT")
                    #elif v['te']>self.intobjs['tecl'].get_info['grid_max']:
                    #    raise ValueError("Required Cl's (%d) for TE are larger than the ones in the emulator (%d)", v['te'], self.intobjs['tecl'].get_info['grid_max'])
                    if 'te' in self._must_provide['Cl'].keys():
                        if self._must_provide['Cl']['te'] < v['te']:
                            self._must_provide['Cl']['te'] = v['te']
                    else:
                        self._must_provide['Cl']['te'] = v['te']
                if 'ee' in v.keys():
                    if not 'EE' in self.emu_objs.keys():
                        raise ValueError("You must provide the Cl's for TT")
                    #elif v['ee']>self.intobjs['eecl'].get_info['grid_max']:
                    #    raise ValueError("Required Cl's (%d) for EE are larger than the ones in the emulator (%d)", v['ee'], self.intobjs['eecl'].get_info['grid_max'])
                    if 'ee' in self._must_provide['Cl'].keys():
                        if self._must_provide['Cl']['ee'] < v['ee']:
                            self._must_provide['Cl']['ee'] = v['ee']
                    else:
                        self._must_provide['Cl']['ee'] = v['ee']
            else:
                self._must_provide[req] = v
                    


                
    def set(self, params_values_dict):
        pass

    def calculate(self, state, want_derived=True, **params_values_dict):
        
        for quantity in self._must_provide.keys():
            if quantity == 'Cl':
                cls = self.compute_Cl(params_values_dict)
                state['Cl'] = cls
            elif quantity == 'Pk':
                pk = self.compute_Pk(params_values_dict)
                state['Pk'] = pk
            else:
                raise ValueError("Unknown quantity %s" % quantity)
            
        # Get Cl's from intobj
        state['derived_extra'] = {'T_cmb': 2.7255}

        return state
                
    def compute_Cl(self, params_values_dict, ell_factor=False, units="FIRASmuK2"):
        # Get Cl's from intobj
        cls = {}
        if 'tt' in self._must_provide['Cl'].keys():
            z, ell, tt = self.get_prediction('TT',params_values_dict)
            cls['tt'] = tt[0]
        if 'te' in self._must_provide['Cl'].keys():
            z, ell, te = self.get_prediction('TE',params_values_dict)
            cls['te'] = te[0]
        if 'ee' in self._must_provide['Cl'].keys():
            z, ell, ee = self.get_prediction('EE',params_values_dict)
            cls['ee'] = ee[0]
        
        # use this when working   
        cls['ell'] = np.array(ell,dtype=int) 

        return cls
    
    def get_Cl(self, ell_factor=False, units="FIRASmuK2", lensed=True):
        
        cls = deepcopy(self.current_state['Cl'])

        # unit conversion and ell_factor
        ells_factor = \
            ((cls["ell"] + 1) * cls["ell"] / (2 * np.pi))[2:] if ell_factor else 1
        units_factor = self._cmb_unit_factor(
            units, self.current_state['derived_extra']['T_cmb'])
        for cl in cls:
            if cl not in ['pp', 'ell']:
                cls[cl][2:] *= units_factor ** 2 * ells_factor
        if lensed and "pp" in cls and ell_factor:
            cls['pp'][2:] *= ells_factor ** 2 * (2 * np.pi)

        return cls
    
    def compute_Pk(self, params_values_dict, z=None, ell_factor=False, units="Mpc"):
        # Get Pk from intobj
        return None
                

