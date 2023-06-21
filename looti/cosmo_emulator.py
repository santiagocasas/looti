from looti import datahandle as dhl
from looti import dictlearn as dcl
import numpy as np

class CosmoEmulator:

    def __init__(self, cosmo_quantities, cosmo_inputs, data_path):
        
        self.cosmo_quantities = cosmo_quantities
        self.cosmo_inputs = cosmo_inputs
        self.data_path = data_path

        self.num_parameters = cosmo_inputs.shape[-1]

        self.intobjs = {}


    def read_data(self, n_train, n_test):

        self.data = {}

        for quant in self.cosmo_quantities:

            data_folder = self.data_path
            datafile_ext = quant
            datafile_ref = quant + '_ref'

            emulation_data = dhl.DataHandle(datafile_ext,
                                            data_folder,
                                            datafile_ref,
                                            num_parameters=self.num_parameters,
                                            data_type=quant,
                                            features_name='grid',
                                            features_to_Log=True,
                                            normalize_by_reference=True,
                                            normalize_by_mean_std=True) 
            emulation_data.read_csv_pandas()
            emulation_data.calculate_ratio_by_redshifts(emulation_data.z_vals)
            emulation_data.calculate_data_split(n_train=n_train,
                                                n_test=n_test,
                                                verbosity=0,
                                                manual_split=True)
            emulation_data.data_split()
            
            self.data[quant] = emulation_data


    def create_intobj(self, cosmo_quantity, ncomp):

        emulation_data = self.data[cosmo_quantity]

        PCAop = dcl.LearningOperator(method='PCA', 
                                     ncomp=ncomp, 
                                     gp_n_rsts=40, 
                                     gp_length=np.ones(self.num_parameters), 
                                     verbosity=0)
        intobj = dcl.LearnData(PCAop)
        intobj.interpolate(train_data=emulation_data.matrix_datalearn_dict['train'],
                           train_samples=emulation_data.train_samples)

        self.intobjs[cosmo_quantity] = intobj    


    def get_prediction(self, cosmo_quantity, input_params, redshift=None):

        if redshift == None:
            params_requested = input_params.reshape(1, -1)
        else:
            params_requested = np.c_[redshift, np.tile(input_params, (len(redshift),1))]

        intobj = self.intobjs[cosmo_quantity]
        emulation_data = self.data[cosmo_quantity]

        predicted = intobj.predict(params_requested)
        prediction_reconstructed = dcl.reconstruct_spectra(ratios_predicted=predicted, 
                                                           emulation_data=emulation_data)

        return prediction_reconstructed[list(prediction_reconstructed.keys())[0]]
    
