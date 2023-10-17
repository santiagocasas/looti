from looti import cosmo_emulator as cem
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time
import os

# path and file name of the training data set (pandas dataframe)
data_path = '../training_data/MnuNur_redshifts/1p0/'
# file_name_list = ['Plin', 'Plincb', 'Pnonlin', 'Pnonlincb'] 
file_name_list=['background_H', 'sigma8', 'D_Growth', 'f_GrowthRate']

# type of observable to be emulated
# cosmo_quantity_list = ['Plin', 'Plincb', 'Pnonlin', 'Pnonlincb']
cosmo_quantity_list = ['background_H', 'sigma8', 'D_Growth', 'f_GrowthRate']

# number of varying parameters in data set
n_params = 7

# set size of training and test data set
n_train_list = [500] # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_test = 100
test_indices=[list(np.random.randint(1, 1100, n_test))]

# choose redshifts
redshifts = [0]
save_name = '_1'

# choose if during training the logarithm of the grid and spectrum should be used
features_to_Log = False
observable_to_Log = False

# choose number of PCA components
ncomp_list = [15]

# option to have a seperate Gaussian process for each PCA component
mult_gp = True


external_info = {'config_yaml': '../readfile_configs/read_input4cast.yaml'}
LootiEmu_pk = cem.CosmoEmulator(external_info=external_info)

def read(file_name, cosmo_quantity, n_train):

    ## DATA---------------------------------------------------------------------
    LootiEmu_pk.read_data(cosmo_quantity=cosmo_quantity, 
                        data_path=data_path, 
                        file_name=file_name, 
                        n_params=n_params, 
                        n_train=n_train, 
                        n_test=n_test,
                        test_indices=test_indices,
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

    # pickle.dump(emulation_data_pk, open(save_emus_dir + '%s_pca%i_data.sav' %(cosmo_quantity, ncomp), 'wb'))
    # pickle.dump(intobj_pk, open(save_emus_dir+ '%s_pca%i.sav' %(cosmo_quantity, ncomp), 'wb')) 

    ## PLOT---------------------------------------------------------------------

    # colors = plt.cm.coolwarm(np.linspace(0, 1, len(emulation_data_pk.test_samples)))
    # fig, ax =plt.subplots(3, figsize=(7, 6))
    # fig.set_tight_layout(tight=True)

    # # get a list of the varying parameters
    # params_varying = list(LootiEmu_pk.get_params(cosmo_quantity=cosmo_quantity).keys())

    # # get true target spectra of the test dataset
    # test_indices = emulation_data_pk.test_splitdict[0]
    # truth_spectrum = emulation_data_pk.df_ext.loc[cosmo_quantity].values[test_indices]

    # max_res = []
    # mean_res = []
    # max_res_cv = []
    # mean_res_cv = []
    # for plot_index, color in enumerate(colors):
    
    #     # create input dictionary with values of test sample for each parameter
    #     input_values = emulation_data_pk.test_samples[plot_index]
    #     input_dict_pk = dict()
    #     for param, value in zip(params_varying, input_values):
    #         input_dict_pk[param] = value

    #     # get the grid and the predicted spectrum for the given input
    #     try:
    #         z_grid, grid_temp, pk_test_list = LootiEmu_pk.get_prediction(cosmo_quantity=cosmo_quantity, input_dict=input_dict_pk)
    #     except:
    #         continue
        
    #     pk_test = pk_test_list[0][2:]
    #     # transform grid in case it is given logarithmically
    #     if features_to_Log==True:
    #         grid = np.power(10, grid_temp)[2:]
    #     else:
    #         grid = grid_temp[2:]


    #     # get the target spectra for current index in test dataset
    #     pk_truth = truth_spectrum[plot_index][2:]
        
    #     # upper plot: target and predicted spectrum 
    #     ax[0].semilogx(grid, pk_truth, c='cornflowerblue', label='truth')
    #     ax[0].semilogx(grid, pk_test, c='firebrick', linestyle='--', label='prediction')

    #     # np.seterr(divide='ignore')
    #     residuals = (pk_test - pk_truth)
    #     max_res.append(np.max(np.abs(residuals)))
    #     mean_res.append(np.mean(np.abs(residuals)))

    #     # np.seterr()
    #     ax[1].semilogx(grid, residuals, color=color)

    # print('Max Residual:', np.mean(max_res))
    # print('Mean Residual:', np.mean(mean_res))
    # # set lables and title
    # ax[0].set_title('%s test data - %i PCA components' %(cosmo_quantity, ncomp))
    # ax[0].set_ylabel('$C_{\ell}^{%s}$' %(cosmo_quantity))
    # ax[0].set_xticklabels([])
    # ax[1].set_ylabel('$\Delta C_{\ell}^{%s}$' %(cosmo_quantity))
    # ax[1].set_xticklabels([])
    # ax[2].set_ylabel('$\Delta C_{\ell}^{%s} / \sigma_{\ell}^{%s}$' %(cosmo_quantity, cosmo_quantity))
    # ax[2].set_xlabel('$k$')

    # plt.tight_layout()
    # plt.savefig(save_plots_dir + '%s_%i.png' %(cosmo_quantity, ncomp))

    # intobj_pk.max_test_residual = np.mean(max_res)
    # intobj_pk.mean_test_residual = np.mean(mean_res)
    # intobj_pk.max_test_residual_cv = np.mean(max_res_cv)
    # intobj_pk.mean_test_residual_cv = np.mean(mean_res_cv)

    pickle.dump(intobj_pk, open(save_emus_dir + cosmo_quantity + save_name+'.sav', 'wb'))

## ---------------------------------------------------------------------------------------------------------


for n_train in n_train_list:
    print("Train on %i samples" %(n_train))
    save_emus_dir = '/work/bk935060/Looti/looti/emulators/MnuNur_5sigma_%i_npca%i/' %(n_train, ncomp_list[0])
    save_plots_dir = '/work/bk935060/Looti/looti/plots/MnuNur_5sigma_%i_npca%i/' %(n_train, ncomp_list[0])

    if not os.path.exists(save_emus_dir):
        os.makedirs(save_emus_dir)
        print("Emuators will be saved in:", save_emus_dir)
    else:
        print("Directory already exists. Adding emulators to:", save_emus_dir)

    if not os.path.exists(save_plots_dir):
        os.makedirs(save_plots_dir)
        print("Plots will be saved in:", save_plots_dir)
    else:
        print("Directory already exists. Adding plots to:", save_plots_dir)

    for file_name, cosmo_quantity in zip(file_name_list, cosmo_quantity_list):

        print('Quantity:', cosmo_quantity)
        read(file_name=file_name, cosmo_quantity=cosmo_quantity, n_train=n_train)

    for file_name, cosmo_quantity in zip(file_name_list, cosmo_quantity_list):
        for ncomp in ncomp_list:

            print('PCA components:', ncomp)
            run_all(cosmo_quantity=cosmo_quantity, ncomp=ncomp)
            print('done')

        print('---------------------------------')