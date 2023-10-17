import os
import numpy as np
import pandas as pd
#import modin.pandas as pd
import yaml
#import modin.config as modin_cfg
import time
#modin_cfg.Engine.put("dask")  # Modin will use Dask


from configparser import ConfigParser
from io import StringIO



class FileReader():

    def __init__(self,path_config_file = "../config_read.yaml", csv_zip=True) :
        self.__read__config(path_config_file)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.csv_zip = csv_zip
        self.fileformat='.csv'
        if self.csv_zip:
            self.fileformat = self.fileformat+'.zip'
    

    def __read__config(self,path_config_file):
        
        with open(path_config_file,'r') as file:
            Param_list = yaml.load(file, Loader=yaml.FullLoader)

        self.folders_path = Param_list["folders_path"]
        self.params_file = Param_list["params_file"]
        self.reference_folder = Param_list["reference_folder"]
        self.save_path = Param_list["save_path"]
        self.save_name = Param_list["save_name"] 

        self.z_file_name = Param_list["z_file_name"]
        self.k_file_name = Param_list["k_file_name"]
        self.ells_file_name = Param_list["ells_file_name"]
        
        self.data_file_names = Param_list["data_file_names"]
        self.data_types = Param_list["data_types"]
        self.training_redshifts = Param_list["training_redshifts"]
        self.n_samples = Param_list['n_samples']
        self.n_training_samples = Param_list['n_training_samples']
        self.n_test_samples = Param_list['n_test_samples']


    def create_dataframes(self):
        self.folders = self.read_folder(self.folders_path)
        self.get_grid()

        for file_name, data_type in zip(self.data_file_names, self.data_types):
            
            print("Creating data frame for : ", data_type)
            save_data_string = self.save_path+self.save_name+'_'+data_type+self.fileformat
            save_ref_string = self.save_path+self.save_name+'_'+data_type+'_ref'+self.fileformat
            grid_param = file_name.split('.')[0][-1]

            if grid_param == 'k':
                df_ref = self.create_k_reference_dataframe(file_name, data_type)
                df_ref.to_csv(save_ref_string)
                df_ext = self.create_k_dataframe(file_name, data_type)
                df_ext.to_csv(save_data_string)
                
            elif grid_param == 'z':
                df_ref = self.create_z_reference_dataframe(file_name, data_type)
                df_ref.to_csv(save_ref_string)
                df_ext = self.create_z_dataframe(file_name, data_type)
                df_ext.to_csv(save_data_string)
            
            elif grid_param == 's':
                df_ref = self.create_ells_reference_dataframe(file_name, data_type)
                df_ref.to_csv(save_ref_string)
                df_ext = self.create_ells_dataframe(file_name, data_type)
                df_ext.to_csv(save_data_string)
            else:
                print("Pattern not found, skipping this quantity: ", data_type)
                continue

            self.params_varying = list(self.read_config(self.folders_path, self.reference_folder, self.params_file).keys())
            print('Number of parameters varying:', len(self.params_varying))
            print('Parameters:', self.params_varying)
            print('Number of samples in dataset:', self.n_samples)
            print('Dataframe saved to:', save_data_string)
            print('Reference dataframe saved to:', save_ref_string)
            print('------------------------------------------')


    def create_k_reference_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        folder = self.reference_folder

        observable = self.read_files(self.folders_path, folder, data_file_name)
        k_array = self.global_k_array
        names = ['data_type', 'redshift']

        values = []
        for zz in self.global_z_array:
            temp = [data_type, zz]
            values.append(temp)

        multiIndex1 = pd.MultiIndex.from_tuples(values, names=names)
        df_temp = pd.DataFrame(data=observable, index=multiIndex1, columns=np.arange(1, len(k_array)+1))
        ## TODO: This concatenation is not needed, or this whole function can be merged with the one below
        dataframe = pd.concat([dataframe, df_temp])
            
        dataframe.loc["grid",:] = k_array
        print(" Reference k-dataframe created ") 
        return dataframe


    def create_k_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        try: 
            k_array = self.global_k_array
        except AttributeError:
            raise AttributeError("global_k_array has to be created first")

        print("Looping over {:d} folders".format(self.n_samples))
        tini = time.time()
        for iif, folder in enumerate(self.folders):
            if iif > self.n_samples:
                print("Maximum number of n_samples reached")
                break

            config_dict = self.read_config(self.folders_path, folder, self.params_file)
            # params_dict = self.create_parameter_dictionary(pars_dict, config_dict)

            observable = self.read_files(self.folders_path, folder, data_file_name)

            names = ['data_type', 'redshift']
            for i, pp in enumerate(config_dict.keys()): ## pars_dict
                names.append('parameter_' + str(i+1))
                names.append('parameter_' + str(i+1) + '_value')

            values = []
            for zz in self.global_z_array:
                temp = [data_type, zz]
                for p, v in zip(config_dict.keys(), config_dict.values()): ## pars_dict, pars_dict
                    temp.append(p)
                    temp.append(v)
                values.append(temp)

            multiIndex1 = pd.MultiIndex.from_tuples(values, names=names)

            columns = np.arange(1, len(k_array)+1)


            df_temp = pd.DataFrame(data=observable, index=multiIndex1, columns=columns)
            dataframe = pd.concat([dataframe, df_temp])
        tfin = time.time()
        print("Time needed for Dataframe creation {:.4f}".format(tfin-tini))
  
        dataframe.loc["grid",:] = k_array
        
        return dataframe
    

    def create_z_reference_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        folder = self.reference_folder

        observable = self.read_files(self.folders_path, folder, data_file_name)
        z_array = self.global_z_array

        names = ['data_type', 'redshift']

        values = [data_type, 0.0]
        multiIndex1 = pd.MultiIndex.from_tuples([values], names=names)
        df_temp = pd.DataFrame(data=[observable], index=multiIndex1, columns=np.arange(1, len(z_array)+1))
        dataframe = pd.concat([dataframe, df_temp])
        print(" Reference z-dataframe created ") 
            
        dataframe.loc["grid",:] = z_array
        
        return dataframe
    

    def create_z_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        try: 
            z_array = self.global_z_array
        except AttributeError:
            raise AttributeError("global_z_array has to be created first")
        print("Looping over {:d} folders".format(self.n_samples))
        for iif, folder in enumerate(self.folders):
            if iif > self.n_samples:
                break
            config_dict = self.read_config(self.folders_path, folder, self.params_file)

            observable = self.read_files(self.folders_path, folder, data_file_name)

            names = ['data_type', 'redshift']
            for i, pp in enumerate(config_dict.keys()):
                names.append('parameter_' + str(i+1))
                names.append('parameter_' + str(i+1) + '_value')

            values = [data_type, 0.0]
            for p, v in zip(config_dict.keys(), config_dict.values()):
                values.append(p)
                values.append(v)
            
            multiIndex1 = pd.MultiIndex.from_tuples([values], names=names)
            df_temp = pd.DataFrame(data=[observable], index=multiIndex1, columns=np.arange(1, len(z_array)+1))
            dataframe = pd.concat([dataframe, df_temp])

  
        dataframe.loc["grid",:] = z_array
        
        return dataframe

    def create_ells_reference_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        folder = self.reference_folder

        observable = self.read_files(self.folders_path, folder, data_file_name)
        ells_array = self.global_ells_array

        names = ['data_type', 'redshift']

        values = [data_type, 0.0]
        multiIndex1 = pd.MultiIndex.from_tuples([values], names=names)
        df_temp = pd.DataFrame(data=[observable], index=multiIndex1, columns=np.arange(1, len(ells_array)+1))
        dataframe = pd.concat([dataframe, df_temp])
        print(" Reference z-dataframe created ") 
            
        dataframe.loc["grid",:] = ells_array
        
        return dataframe
    

    def create_ells_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        try: 
            ells_array = self.global_ells_array
        except AttributeError:
            raise AttributeError("global_ells_array has to be created first")
        print("Looping over {:d} folders".format(self.n_samples))
        for iif, folder in enumerate(self.folders):
            if iif > self.n_samples:
                break
            config_dict = self.read_config(self.folders_path, folder, self.params_file)

            observable = self.read_files(self.folders_path, folder,  data_file_name)

            names = ['data_type', 'redshift']
            for i, pp in enumerate(config_dict.keys()):
                names.append('parameter_' + str(i+1))
                names.append('parameter_' + str(i+1) + '_value')

            values = [data_type, 0.0]
            for p, v in zip(config_dict.keys(), config_dict.values()):
                values.append(p)
                values.append(v)
            
            multiIndex1 = pd.MultiIndex.from_tuples([values], names=names)
            df_temp = pd.DataFrame(data=[observable], index=multiIndex1, columns=np.arange(1, len(ells_array)+1))
            dataframe = pd.concat([dataframe, df_temp])

  
        dataframe.loc["grid",:] = ells_array
        
        return dataframe

    def read_folder(self, folder_path):
        # folders = os.listdir(path + folder)
        folders = None
        if os.path.isdir(folder_path):
            folders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
            folders.remove(self.reference_folder)
            self.n_folders = len(folders)
        return folders



    def read_config(self, folder_path, config_folder, params_file):
        with open(folder_path + config_folder + "/" + params_file) as f:
            config = StringIO()
            config.write('[dummy_section]\n')
            config.write(f.read().replace('%', '%%'))
            config.seek(0, os.SEEK_SET)

            cp = ConfigParser()
            cp.optionxform=str
            cp.read_file(config)

            return dict(cp.items('dummy_section'))

    def get_grid(self):

        path = self.folders_path + self.reference_folder + "/" 

        try:
            self.global_z_array = np.load(path + self.z_file_name)['arr_0']
        except:
            pass
        try:
            self.global_k_array = np.load(path + self.k_file_name)['arr_0']
        except:
            pass
        try: 
            self.global_ells_array = np.load(path + self.ells_file_name)['arr_0']
        except:
            pass
        

    def read_files(self, folders_path, folder_name, data_file_name):

        observable  = np.load(folders_path + folder_name + "/" + data_file_name)['arr_0']

        return observable
    

    def filter_redshift(self, save_path, redshift=0, n_params=None):

        if n_params == None:
            n_params = len(self.params_varying)

        n_index = 2 + 2 * n_params

        for data_type in self.data_types:

            df_all = pd.read_csv(self.save_path+self.save_name+'_'+data_type+self.fileformat, index_col=list(range(n_index)))
            df_all_ref = pd.read_csv(self.save_path+self.save_name+'_'+data_type+'_ref'+self.fileformat, index_col=list(range(2)))

            df_data = df_all[df_all.index.get_level_values('redshift')==redshift]
            df_data_ref = df_all_ref[df_all_ref.index.get_level_values('redshift')==redshift]

            df_grid = df_all[df_all.index.get_level_values('data_type')=='grid']
            df_grid_ref = df_all_ref[df_all_ref.index.get_level_values('data_type')=='grid']

            df_z0 = pd.concat([df_data, df_grid])
            df_z0_ref = pd.concat([df_data_ref, df_grid_ref])

            df_z0.to_csv(save_path+data_type+self.fileformat)
            df_z0_ref.to_csv(save_path+data_type+'_ref'+self.fileformat)

            print('%s at redshift %f saved under %s' %(data_type, redshift, save_path+data_type+self.fileformat))
            print('Reference %s at redshift %f saved under %s' %(data_type, redshift, save_path+data_type+'_ref'+self.fileformat))
