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

        self.main_dir = Param_list["main_dir"]
        self.config_file = Param_list["config_file"]
        self.folders_path = Param_list["folders_path"]
        self.params_file = Param_list["params_file"]
        self.reference_folder = Param_list["reference_folder"]

        self.z_file_name = Param_list["z_file_name"]
        self.k_file_name = Param_list["k_file_name"]
        self.data_file_name = Param_list["data_file_name"]


    def create_reference_dataframe(self):
        pars_dict = self.read_main(self.main_dir, self.config_file)
        folders = self.read_folder(self.main_dir, self.folders_path)

        dataframe = pd.DataFrame()
        folder = self.reference_folder

        config_dict = self.read_config(self.main_dir, self.folders_path, folder, self.params_file)
        params_dict = self.create_parameter_dictionary(pars_dict, config_dict)

        z_array, k_array, Plin_kz = self.read_files(self.main_dir, self.folders_path, folder)

        names = ['data_type', 'redshift']

        values = []
        for zz in z_array:
            temp = ['Plin', zz]
            values.append(temp)

        multiIndex1 = pd.MultiIndex.from_tuples(values, names=names)
        df_temp = pd.DataFrame(data=Plin_kz, index=multiIndex1, columns=np.arange(1, len(k_array)+1))
        dataframe = pd.concat([dataframe, df_temp])
            
        dataframe.loc["k_grid",:] = k_array
        
        return dataframe


    def create_dataframe(self):
        pars_dict = self.read_main(self.main_dir, self.config_file)
        folders = self.read_folder(self.main_dir, self.folders_path)

        dataframe = pd.DataFrame()
        for folder in folders:

            config_dict = self.read_config(self.main_dir, self.folders_path, folder, self.params_file)
            params_dict = self.create_parameter_dictionary(pars_dict, config_dict)

            z_array, k_array, Plin_kz = self.read_files(self.main_dir, self.folders_path, folder)

            names = ['data_type', 'redshift']
            for i, pp in enumerate(pars_dict.keys()):
                names.append('parameter_' + str(i+1))
                names.append('parameter_' + str(i+1) + '_value')

            values = []
            for zz in z_array:
                temp = ['Plin', zz]
                for p, v in zip(params_dict.keys(), params_dict.values()):
                    temp.append(p)
                    temp.append(v)
                values.append(temp)

            multiIndex1 = pd.MultiIndex.from_tuples(values, names=names)
            df_temp = pd.DataFrame(data=Plin_kz, index=multiIndex1, columns=np.arange(1, len(k_array)+1))
            dataframe = pd.concat([dataframe, df_temp])
            
        dataframe.loc["k_grid",:] = k_array
        
        return dataframe


    def read_folder(self, path, folder):
        folders = os.listdir(path + folder)
        folders.remove(self.reference_folder)
        return folders


    def read_main(self, config_dir, config_file):
        configmain = ConfigParser()
        configmain.optionxform=str
        configmain.read(config_dir + config_file)
        pars_var_dict = dict(configmain.items('params_varying'))
        return pars_var_dict


    def read_config(self, main_dir, config_dir, config_folder, config_file):
        with open(main_dir + config_dir + config_folder + "/" + config_file) as f:
            config = StringIO()
            config.write('[dummy_section]\n')
            config.write(f.read().replace('%', '%%'))
            config.seek(0, os.SEEK_SET)

            cp = ConfigParser()
            cp.optionxform=str
            cp.read_file(config)

            return dict(cp.items('dummy_section'))
        

    def create_parameter_dictionary(self, pars_dict, config_dict):
        pars_var_dict = {}
        for pp in pars_dict.keys():
            if pp == 'omega_m':
                pars_var_dict[pp] = float(config_dict['omega_baryon']) + float(config_dict['omega_cdm']) + float(config_dict['omega_neutrino'])
            else:
                pars_var_dict[pp] = float(config_dict[pp])

        return pars_var_dict


    def read_files(self, main_dir, folders_path, folder_name, z_file_name='z_values_list.txt', k_file_name='k_values_list.txt', Plin_file_name='Plin-zk.txt'):

        z_array = np.loadtxt(main_dir + folders_path + folder_name + "/" + z_file_name)
        k_array = np.loadtxt(main_dir + folders_path + folder_name + "/" + k_file_name)
        Pks  = np.loadtxt(main_dir + folders_path + folder_name + "/" + Plin_file_name)

        return z_array, k_array, Pks
