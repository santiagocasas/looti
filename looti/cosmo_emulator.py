from looti import datahandle as dhl
from looti import dictlearn as dcl
import numpy as np
import pickle

class CosmoEmulator:

    def __init__(self):

        self.data = {}
        self.intobjs = {}


    def read_data(self, cosmo_quantity, data_path, n_params, n_train, n_test, **kwargs):

        data_folder = data_path
        datafile_ext = cosmo_quantity
        datafile_ref = cosmo_quantity + '_ref'

        emulation_data = dhl.DataHandle(datafile_ext,
                                        data_folder,
                                        datafile_ref,
                                        num_parameters=n_params,
                                        data_type=cosmo_quantity,
                                        features_name=kwargs.get('features_name', 'grid'),
                                        features_to_Log=kwargs.get('features_to_Log', True),
                                        normalize_by_reference=True,
                                        normalize_by_mean_std=True) 
        emulation_data.read_csv_pandas()
        emulation_data.calculate_ratio_by_redshifts(emulation_data.z_vals)
        emulation_data.calculate_data_split(n_train=n_train,
                                            n_test=n_test,
                                            verbosity=0,
                                            manual_split=True)
        emulation_data.data_split(verbosity=0)
        
        self.data[cosmo_quantity] = emulation_data


    def read_intobj(self, cosmo_quantity, data_path, intobj_path):

        emulation_data = pickle.load(open(data_path, 'rb'))
        self.data[cosmo_quantity] = emulation_data

        intobj = pickle.load(open(intobj_path, 'rb'))
        self.intobjs[cosmo_quantity] = intobj


    def create_intobj(self, cosmo_quantity, n_params, **kwargs):

        emulation_data = self.data[cosmo_quantity]

        PCAop = dcl.LearningOperator(method='PCA', 
                                     ncomp=kwargs.get('ncomp', 8), 
                                     gp_n_rsts=kwargs.get('gp_n_rsts', 40), 
                                     gp_length=kwargs.get('gp_length', np.ones(n_params)), 
                                     verbosity=0)
        intobj = dcl.LearnData(PCAop)
        intobj.interpolate(train_data=emulation_data.matrix_datalearn_dict['train'],
                           train_samples=emulation_data.train_samples)

        self.intobjs[cosmo_quantity] = intobj    


    def get_prediction(self, cosmo_quantity, input_dict, redshift=None):

        emulation_data = self.data[cosmo_quantity]
        intobj = self.intobjs[cosmo_quantity]

        input_params = self.read_input_dict(input_dict, emulation_data)

        if redshift == None:
            params_requested = input_params.reshape(1, -1)
        else:
            params_requested = np.c_[redshift, np.tile(input_params, (len(redshift),1))]

        predicted = intobj.predict(params_requested)
        prediction_reconstructed = dcl.reconstruct_spectra(ratios_predicted=predicted, 
                                                           emulation_data=emulation_data)
        
        fgrid = emulation_data.fgrid

        return fgrid, prediction_reconstructed[list(prediction_reconstructed.keys())[0]]
    

    def read_input_dict(self, input_dict, emulation_data):

        input_params_list = []
        for param in emulation_data.paramnames_dict.values():
            try:
                input_params_list.append(input_dict[param])
            except KeyError:
                print('Missing input value for parameter:', param)
        input_params = np.array(input_params_list)

        return input_params
    

    def get_params(self, cosmo_quantity):

        emulation_data = self.data[cosmo_quantity]
        params = list(emulation_data.paramnames_dict.values())

        return params
    

    def get_info(self, cosmo_quantity):

        info_dict = {}
        info_dict['grid_max'] = self.data[cosmo_quantity].fgrid.max()
        info_dict['grid_min'] = self.data[cosmo_quantity].fgrid.max()
        ## TODO: add whatever info we need

        return info_dict