from looti import datahandle as dhl
from looti import dictlearn as dcl
import numpy as np
import pickle

class CosmoEmulator:

    def __init__(self):

        self.data = {}
        self.emu_objs = {}


    def read_data(self, cosmo_quantity, data_path, file_name, n_params, n_train, n_test, **kwargs):

        data_folder = data_path
        datafile_ext = file_name
        datafile_ref = file_name + '_ref'

        emulation_data = dhl.DataHandle(extmodel_filename=datafile_ext,
                                        data_dir=data_folder,
                                        refmodel_filename=datafile_ref,
                                        num_parameters=n_params,
                                        data_type=cosmo_quantity,
                                        features_name=kwargs.get('features_name', 'grid'),
                                        features_to_Log=kwargs.get('features_to_Log', True),
                                        observable_to_Log=kwargs.get('observable_to_Log', False),
                                        normalize_by_reference=True,
                                        normalize_by_mean_std=True) 
        emulation_data.read_csv_pandas()
        emulation_data.calculate_ratio_by_redshifts(emulation_data.z_vals)
        emulation_data.calculate_data_split(n_train=n_train,
                                            n_test=n_test,
                                            verbosity=0,
                                            manual_split=True,
                                            train_redshift_indices=kwargs.get('redshifts', [0]),
                                            test_redshift_indices=kwargs.get('redshifts', [0]))
        emulation_data.data_split(verbosity=0)
        
        self.data[cosmo_quantity] = emulation_data


    def read_emulator(self, cosmo_quantity, data_path, emulator_path):

        emulation_data = pickle.load(open(data_path, 'rb'))
        self.data[cosmo_quantity] = emulation_data

        emuobj = pickle.load(open(emulator_path, 'rb'))
        self.emu_objs[cosmo_quantity] = emuobj


    def create_emulator(self, cosmo_quantity, n_params, **kwargs):

        emulation_data = self.data[cosmo_quantity]

        PCAop = dcl.LearningOperator(method='PCA',
                                     mult_gp=kwargs.get('mult_gp', False),
                                     ncomp=kwargs.get('ncomp', 8),
                                     gp_n_rsts=kwargs.get('gp_n_rsts', 40),
                                     gp_length=kwargs.get('gp_length', np.ones(n_params)),
                                     verbosity=0)
        emuobj = dcl.LearnData(PCAop)
        emuobj.interpolate(train_data=emulation_data.matrix_datalearn_dict['train'],
                           train_samples=emulation_data.train_samples)

        self.emu_objs[cosmo_quantity] = emuobj


    def get_prediction(self, cosmo_quantity, input_dict, redshift=None):

        emulation_data = self.data[cosmo_quantity]
        emuobj = self.emu_objs[cosmo_quantity]

        input_params = self.read_input_dict(input_dict, cosmo_quantity)

        if redshift == None:
            params_requested = input_params.reshape(1, -1)
        else:
            params_requested = np.c_[redshift, np.tile(input_params, (len(redshift),1))]

        predicted = emuobj.predict(params_requested)
        prediction_reconstructed = dcl.reconstruct_spectra(ratios_predicted=predicted, 
                                                           emulation_data=emulation_data,
                                                           normalization=True,
                                                           observable_to_Log=emulation_data.observable_to_Log)
        
        fgrid = emulation_data.fgrid

        return fgrid, prediction_reconstructed[list(prediction_reconstructed.keys())[0]]
    

    def read_input_dict(self, input_dict, cosmo_quantity):

        limits_dict = self.get_params(cosmo_quantity)
        paramnames_dict = self.data[cosmo_quantity].paramnames_dict

        input_params_list = []
        for param in limits_dict.keys():
            try:
                if input_dict[param] >= limits_dict[param][0] and input_dict[param] <= limits_dict[param][1]:
                    input_params_list.append(input_dict[param])
                else:
                    raise ValueError('Requested value for %s is outside of training region.' %param)
            except KeyError:
                print('Missing input value for parameter:', param)
        input_params = np.array(input_params_list)

        return input_params
    

    def get_params(self, cosmo_quantity):

        emulation_data = self.data[cosmo_quantity]
        if emulation_data.multiple_z:
            paramnames = ['redshift'] + list(emulation_data.paramnames_dict.values())
        else:
            paramnames = list(emulation_data.paramnames_dict.values())

        params_dict = {}
        for ii, param in enumerate(paramnames):
            params_dict[param] = (emulation_data.train_samples[:,ii].min(), emulation_data.train_samples[:,ii].max())

        return params_dict
    

    def get_info(self, cosmo_quantity):

        info_dict = {}
        info_dict['grid_max'] = self.data[cosmo_quantity].fgrid.max()
        info_dict['grid_min'] = self.data[cosmo_quantity].fgrid.min()
        ## TODO: add whatever info we need

        return info_dict