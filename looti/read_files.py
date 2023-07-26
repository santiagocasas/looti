import os
import numpy as np
import pandas as pd
import yaml

from configparser import ConfigParser
from io import StringIO



class FrameConstructor():

    def __init__(self,path_config_file = "../config_read.yaml") :
       self.__read__config(path_config_file)        


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
        self.data_file_names = Param_list["data_file_names"]
        self.data_types = Param_list["data_types"]


    def create_dataframes(self):

        for file_name, data_type in zip(self.data_file_names, self.data_types):

            save_data_string = self.save_path+self.save_name+'_'+data_type+'.csv'
            save_ref_string = self.save_path+self.save_name+'_'+data_type+'_ref.csv'
            grid_param = file_name.split('.')[0][-1]

            if grid_param == 'k':
                df_ext = self.create_k_dataframe(file_name, data_type)
                df_ext.to_csv(save_data_string)
                df_ref = self.create_k_reference_dataframe(file_name, data_type)
                df_ref.to_csv(save_ref_string)
                
            elif grid_param == 'z':
                df_ext = self.create_z_dataframe(file_name, data_type)
                df_ext.to_csv(save_data_string)
                df_ref = self.create_z_reference_dataframe(file_name, data_type)
                df_ref.to_csv(save_ref_string)

            self.params_varying = list(self.read_config(self.folders_path, self.reference_folder, self.params_file).keys())
            print(data_type)
            print('Number of parameters varying:', len(self.params_varying))
            print('Parameters:', self.params_varying)
            print('Number of samples in dataset:', self.n_samples)
            print('Dataframe saved to:', save_data_string)
            print('Reference dataframe saved to:', save_ref_string)
            print('------------------------------------------')


    def create_k_reference_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        folder = self.reference_folder

        z_array, k_array, observable = self.read_files(self.folders_path, folder, self.z_file_name, self.k_file_name, data_file_name)

        names = ['data_type', 'redshift']

        values = []
        for zz in z_array:
            temp = [data_type, zz]
            values.append(temp)

        multiIndex1 = pd.MultiIndex.from_tuples(values, names=names)
        df_temp = pd.DataFrame(data=observable, index=multiIndex1, columns=np.arange(1, len(k_array)+1))
        dataframe = pd.concat([dataframe, df_temp])
            
        dataframe.loc["grid",:] = k_array
        
        return dataframe


    def create_k_dataframe(self, data_file_name, data_type):
        folders = self.read_folder(self.folders_path)

        dataframe = pd.DataFrame()
        for folder in folders:

            config_dict = self.read_config(self.folders_path, folder, self.params_file)
            # params_dict = self.create_parameter_dictionary(pars_dict, config_dict)

            z_array, k_array, observable = self.read_files(self.folders_path, folder,  self.z_file_name, self.k_file_name, data_file_name)

            names = ['data_type', 'redshift']
            for i, pp in enumerate(config_dict.keys()): ## pars_dict
                names.append('parameter_' + str(i+1))
                names.append('parameter_' + str(i+1) + '_value')

            values = []
            for zz in z_array:
                temp = [data_type, zz]
                for p, v in zip(config_dict.keys(), config_dict.values()): ## pars_dict, pars_dict
                    temp.append(p)
                    temp.append(v)
                values.append(temp)

            multiIndex1 = pd.MultiIndex.from_tuples(values, names=names)

            columns = np.arange(1, len(k_array)+1)


            df_temp = pd.DataFrame(data=observable, index=multiIndex1, columns=columns)
            dataframe = pd.concat([dataframe, df_temp])

  
        dataframe.loc["grid",:] = k_array
        
        return dataframe
    

    def create_z_reference_dataframe(self, data_file_name, data_type):

        dataframe = pd.DataFrame()
        folder = self.reference_folder

        z_array, k_array, observable = self.read_files(self.folders_path, folder, self.z_file_name, self.k_file_name, data_file_name)

        names = ['data_type', 'redshift']

        values = [data_type, 0.0]
        multiIndex1 = pd.MultiIndex.from_tuples([values], names=names)
        df_temp = pd.DataFrame(data=[observable], index=multiIndex1, columns=np.arange(1, len(z_array)+1))
        dataframe = pd.concat([dataframe, df_temp])
            
        dataframe.loc["grid",:] = z_array
        
        return dataframe
    

    def create_z_dataframe(self, data_file_name, data_type):
        folders = self.read_folder(self.folders_path)

        dataframe = pd.DataFrame()
        for folder in folders:

            config_dict = self.read_config(self.folders_path, folder, self.params_file)

            z_array, k_array, observable = self.read_files(self.folders_path, folder,  self.z_file_name, self.k_file_name, data_file_name)

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


    def read_folder(self, folder_path):
        # folders = os.listdir(path + folder)
        folders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
        folders.remove(self.reference_folder)
        self.n_samples = len(folders)
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
        

    # def create_parameter_dictionary(self, pars_dict, config_dict):
    #     pars_var_dict = {}
    #     for pp in pars_dict.keys():
    #         if pp == 'omega_m':
    #             pars_var_dict[pp] = float(config_dict['omega_baryon']) + float(config_dict['omega_cdm']) + float(config_dict['omega_neutrino'])
    #         else:
    #             pars_var_dict[pp] = float(config_dict[pp])

    #     return pars_var_dict


    def read_files(self, folders_path, folder_name, z_file_name, k_file_name, data_file_name):

        z_array = np.loadtxt(folders_path + folder_name + "/" + z_file_name)
        k_array = np.loadtxt(folders_path + folder_name + "/" + k_file_name)
        observable  = np.loadtxt(folders_path + folder_name + "/" + data_file_name)

        return z_array, k_array, observable
    

    def filter_redshift(self, save_path, redshift=0, n_params=None):

        if n_params == None:
            n_params = len(self.params_varying)

        n_index = 2 + 2 * n_params

        for data_type in self.data_types:

            df_all = pd.read_csv(self.save_path+self.save_name+'_'+data_type+'.csv', index_col=list(range(n_index)))
            df_all_ref = pd.read_csv(self.save_path+self.save_name+'_'+data_type+'_ref.csv', index_col=list(range(2)))

            df_data = df_all[df_all.index.get_level_values('redshift')==redshift]
            df_data_ref = df_all_ref[df_all_ref.index.get_level_values('redshift')==redshift]

            df_grid = df_all[df_all.index.get_level_values('data_type')=='grid']
            df_grid_ref = df_all_ref[df_all_ref.index.get_level_values('data_type')=='grid']

            df_z0 = pd.concat([df_data, df_grid])
            df_z0_ref = pd.concat([df_data_ref, df_grid_ref])

            df_z0.to_csv(save_path+data_type+'.csv')
            df_z0_ref.to_csv(save_path+data_type+'_ref.csv')

            print('%s at redshift %f saved under %s' %(data_type, redshift, save_path+data_type+'.csv'))
            print('Reference %s at redshift %f saved under %s' %(data_type, redshift, save_path+data_type+'_ref.csv'))
