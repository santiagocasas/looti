from looti import cosmo_emulator as cem
import numpy as np
import pickle

# path and file name of the training data set (pandas dataframe)
data_path = '../training_data/class_Asw0wa_DP_hypel_GCsp/class_Asw0wa_DP_hypel_z1p65/'
save_path = '../emulators/GCsp'
save_name = '_4'

# number of varying parameters in data set
n_params = 7

# set size of training and test data set
n_train = 575
n_test = 10

# choose redshifts
redshifts = [0]

# choose if during training the logarithm of the grid and spectrum should be used
features_to_Log = False
observable_to_Log = False

# choose number of PCA components
ncomp = 20

# option to have a seperate Gaussian process for each PCA component
mult_gp = True


quantities = ['Plin', 'Pnonlin', 'f_GrowthRate', 'D_Growth']
for cosmo_quantity in quantities:

    LootiEmu_pk = cem.CosmoEmulator()

    print('Reading data for', cosmo_quantity)
    LootiEmu_pk.read_data(cosmo_quantity=cosmo_quantity, 
                        data_path=data_path, 
                        file_name=cosmo_quantity, 
                        n_params=n_params, 
                        n_train=n_train, 
                        n_test=n_test,
                        redshifts=redshifts,
                        features_to_Log=features_to_Log,
                        observable_to_Log=observable_to_Log)

    print('Training emulator for', cosmo_quantity)
    LootiEmu_pk.create_emulator(cosmo_quantity=cosmo_quantity, n_params=n_params, mult_gp=mult_gp, ncomp=ncomp)
    print('Training done')

    emulation_data_pk = LootiEmu_pk.data[cosmo_quantity]
    intobj_pk = LootiEmu_pk.intobjs[cosmo_quantity]

    pickle.dump(intobj_pk, open(save_path + cosmo_quantity + save_name + '.sav', 'wb'))
    pickle.dump(emulation_data_pk, open(save_path + cosmo_quantity + save_name + '_data.sav', 'wb'))
    print('Emulator saved under', save_path + cosmo_quantity + save_name + '.sav')
    print('Data saved under', save_path + cosmo_quantity + save_name + '_data.sav')