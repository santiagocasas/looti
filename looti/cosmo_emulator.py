from looti import datahandle as dhl
from looti import dictlearn as dcl
from looti import read_files as rf
import numpy as np
import pickle
from glob import glob 

class CosmoEmulator:

    def __init__(self, external_info=dict()):

        self.data = {}
        self.emu_objs = {}
        self.external_info = external_info
        self.config_yaml = self.external_info['config_yaml']
        self.FileReader = rf.FileReader(path_config_file=self.config_yaml)

    def create_dataframes(self):
        self.FileReader.create_dataframes()
        self.params_varying = self.FileReader.params_varying
        self.n_params = len(self.params_varying)
        self.data_path = self.FileReader.save_path
        self.data_types = self.FileReader.data_types
        self.n_samples = self.FileReader.n_samples
        self.n_training_samples = self.FileReader.n_training_samples
        self.n_test_samples = self.FileReader.n_test_samples

    
    def read_dataframes(self):
        #for quantity in zip(self.data_types, self.data_path)
        return None


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


    def read_emulator(self, cosmo_quantity, directory):
        emuobj_list = []
        quant_file_list = glob(directory + cosmo_quantity + '*' + '.sav')
        quant_file_list.sort()
        for fi in quant_file_list:
            emuobj = pickle.load(open(fi, 'rb'))
            emuobj = self.add_external_attributes(obje=emuobj, cosmo_quantity=cosmo_quantity)
            emuobj_list.append(emuobj)
        self.emu_objs[cosmo_quantity] = emuobj_list

    def add_external_attributes(self, obje, cosmo_quantity):
        if 'features_to_Log' in self.external_info:
            if cosmo_quantity != 'background_H' and cosmo_quantity != 'sigma8':
                obje.features_to_Log = self.external_info['features_to_Log']
            else:
                obje.features_to_Log = False
        return obje

    def create_emulator(self, cosmo_quantity, n_params, **kwargs):

        emulation_data = self.data[cosmo_quantity]
        gp_length = n_params
        if emulation_data.multiple_z:
            gp_length += 1
        PCAop = dcl.LearningOperator(method='PCA',
                                     mult_gp=kwargs.get('mult_gp', False),
                                     ncomp=kwargs.get('ncomp', 8),
                                     gp_n_rsts=kwargs.get('gp_n_rsts', 40),
                                     gp_length=kwargs.get('gp_length', np.ones(gp_length)),
                                     verbosity=0)
        emuobj = dcl.LearnData(PCAop)
        emuobj.interpolate(emulation_data = emulation_data)
        ## TODO: THIS needs to be changed to accept redshifts
        self.emu_objs[cosmo_quantity] = emuobj


    def get_prediction(self, cosmo_quantity, input_dict, redshift_argument=None):

        emuobj_list = self.emu_objs[cosmo_quantity]

        params_requested = self.read_input_dict(input_dict, cosmo_quantity, 
                                            redshift_argument=redshift_argument)
        
        z_values = []
        predictions_list = []
        fgrid = self.get_fgrid(cosmo_quantity=cosmo_quantity)
        for emuobj in emuobj_list:
            predicted = emuobj.predict(params_requested)
            prediction_reconstructed = emuobj.reconstruct_spectra(ratios_predicted=predicted)
            params_req_tuple = tuple(params_requested[0])
            predictions_list.append(prediction_reconstructed[params_req_tuple])    
            z_values.append(emuobj.z_requested[0])

        return z_values, fgrid, predictions_list
    

    def read_input_dict(self, input_dict, cosmo_quantity, redshift_argument=None):

        limits_dict = self.get_params(cosmo_quantity)

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
        
        if redshift_argument == None:
            input_params = input_params.reshape(1, -1)
        else:
            input_params = np.c_[redshift_argument, np.tile(input_params, (len(redshift_argument),1))]

        return input_params


    def get_fgrid(self, cosmo_quantity):

        emuobj = self.emu_objs[cosmo_quantity][0]
        fgrid = emuobj.fgrid
        if emuobj.features_to_Log:
            fgrid = np.power(10, fgrid)

        return fgrid


    def get_params(self, cosmo_quantity):

        emuobj = self.emu_objs[cosmo_quantity][0]
        params_dict = emuobj.params_dict

        return params_dict
    
    def get_main_ini_dict(self, cosmo_quantity):

        emuobj = self.emu_objs[cosmo_quantity][0]
        main_ini_dict = emuobj.main_dict

        return main_ini_dict
    

    def get_info(self, cosmo_quantity):

        info_dict = {}
        fgrid = self.get_fgrid(cosmo_quantity=cosmo_quantity)
        info_dict['grid_max'] = fgrid.max()
        info_dict['grid_min'] = fgrid.min()
        ## TODO: add whatever info we need

        return info_dict