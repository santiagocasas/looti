from looti import cosmo_emulator as cem
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time

# path and file name of the training data set (pandas dataframe)
data_path = '../training_data/cmb/'
file_name_list = ['cmb_TT', 'cmb_TE', 'cmb_EE']
# file_name_list = ['class_Asw0wa_DP_cmb_2K_Pnonlin', 'class_Asw0wa_DP_cmb_2K_f_GrowthRate', 'class_Asw0wa_DP_cmb_2K_D_Growth', 'class_Asw0wa_DP_cmb_2K_background_H']

# type of observable to be emulated
cosmo_quantity_list = ['TT', 'TE', 'EE']
# cosmo_quantity_list = ['Pnonlin', 'f_GrowthRate', 'D_Growth']

# number of varying parameters in data set
n_params = 7

# set size of training and test data set
n_train = 1000
n_test = 100

# choose redshifts
redshifts = [0]

# choose if during training the logarithm of the grid and spectrum should be used
features_to_Log = False
observable_to_Log = False

# choose number of PCA components
ncomp_list = [20]

# option to have a seperate Gaussian process for each PCA component
mult_gp = True


external_info = {'config_yaml': '../readfile_configs/read_input4cast.yaml'}
LootiEmu_pk = cem.CosmoEmulator(external_info=external_info)

def read(file_name, cosmo_quantity):

    ## DATA---------------------------------------------------------------------
    LootiEmu_pk.read_data(cosmo_quantity=cosmo_quantity, 
                        data_path=data_path, 
                        file_name=file_name, 
                        n_params=n_params, 
                        n_train=n_train, 
                        n_test=n_test,
                        redshifts=redshifts,
                        features_to_Log=features_to_Log,
                        observable_to_Log=observable_to_Log)


def run_all(cosmo_quantity, ncomp):
    
    ## TRAINING-----------------------------------------------------------------
    t_ini = time.time()
    LootiEmu_pk.create_emulator(cosmo_quantity=cosmo_quantity, 
                                n_params=n_params, 
                                mult_gp=mult_gp, 
                                ncomp=ncomp)
    t_fin = time.time() - t_ini
    print('Training time for %s on %i pca components: %.4f' %(cosmo_quantity, ncomp, t_fin))

    emulation_data_pk = LootiEmu_pk.data[cosmo_quantity]
    intobj_pk = LootiEmu_pk.emu_objs[cosmo_quantity][0]

    ## PLOT---------------------------------------------------------------------

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(emulation_data_pk.test_samples)))
    fig, ax =plt.subplots(3, figsize=(7, 6))
    fig.set_tight_layout(tight=True)

    # get a list of the varying parameters
    params_varying = list(LootiEmu_pk.get_params(cosmo_quantity=cosmo_quantity).keys())

    # get true target spectra of the test dataset
    test_indices = emulation_data_pk.test_splitdict[0]
    truth_spectrum = emulation_data_pk.df_ext.loc[cosmo_quantity].values[test_indices]

    for plot_index, color in enumerate(colors):
    
        # create input dictionary with values of test sample for each parameter
        input_values = emulation_data_pk.test_samples[plot_index]
        input_dict_pk = dict()
        for param, value in zip(params_varying, input_values):
            input_dict_pk[param] = value

        # get the grid and the predicted spectrum for the given input
        try:
            z_grid, grid_temp, pk_test_list = LootiEmu_pk.get_prediction(cosmo_quantity=cosmo_quantity, input_dict=input_dict_pk)
        except:
            continue
        
        pk_test = pk_test_list[0]
        # transform grid in case it is given logarithmically
        if features_to_Log==True:
            grid = np.power(10, grid_temp)
        else:
            grid = grid_temp

        # get the target spectra for current index in test dataset
        pk_truth = truth_spectrum[plot_index]
        cv = pk_truth / np.sqrt(grid + 0.5)
        # upper plot: target and predicted spectrum 
        ax[0].plot(grid, pk_truth / cv, c='cornflowerblue', label='truth')
        ax[0].plot(grid, pk_test / cv, c='firebrick', linestyle='--', label='prediction')

        # middle plot: relative residuals
        np.seterr(divide='ignore')
        residuals = 1 - pk_test / truth_spectrum[plot_index]
        np.seterr()
        ax[1].plot(grid, residuals, color=color)

        # lower plot: absolute residuals
        residuals = pk_test - truth_spectrum[plot_index]
        ax[2].plot(grid, residuals, color=color)

    # set lables and title
    ax[0].set_title('%s test data - %i PCA components' %(cosmo_quantity, ncomp))
    ax[0].set_ylabel('Spectra')
    ax[0].set_xticklabels([])
    ax[1].set_ylabel('Relative Residuals')
    ax[1].set_xticklabels([])
    ax[2].set_xlabel(r'$k$')
    ax[2].set_ylabel('Residuals')

    plt.tight_layout()
    plt.savefig('../plots/cmb_new/%s_%i.png' %(cosmo_quantity, ncomp))

    pickle.dump(emulation_data_pk, open('../emulators/cmb_new/%s_pca%i_data.sav' %(cosmo_quantity, ncomp), 'wb'))
    pickle.dump(intobj_pk, open('../emulators/cmb_new/%s_pca%i.sav' %(cosmo_quantity, ncomp), 'wb')) 

for file_name, cosmo_quantity in zip(file_name_list, cosmo_quantity_list):

    print(cosmo_quantity)
    read(file_name=file_name, cosmo_quantity=cosmo_quantity)

    for ncomp in ncomp_list:

        print(ncomp)
        run_all(cosmo_quantity=cosmo_quantity, ncomp=ncomp)
        print('done')

    print('---------------------------------')