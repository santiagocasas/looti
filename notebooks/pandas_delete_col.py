import pandas as pd
import os

load_path = '../training_data/w0wa_sig8/'
files = ['_Plin.csv.zip', '_Plincb.csv.zip', '_Pnonlin.csv.zip', '_Pnonlincb.csv.zip', 
         '_background_H.csv.zip', '_sigma8.csv.zip', '_D_Growth.csv.zip', '_f_GrowthRate.csv.zip']
         # '_TT.csv.zip', '_TE.csv.zip', '_EE.csv.zip']
ref_files = ['_Plin_ref.csv.zip', '_Plincb_ref.csv.zip', '_Pnonlin_ref.csv.zip', '_Pnonlincb_ref.csv.zip', 
         '_background_H_ref.csv.zip', '_sigma8_ref.csv.zip', '_D_Growth_ref.csv.zip', '_f_GrowthRate_ref.csv.zip']
         # '_TT_ref.csv.zip', '_TE_ref.csv.zip', '_EE_ref.csv.zip']

save_path = '../training_data/w0wa_nosig8/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in files:
    print(file)
    df = pd.read_csv(load_path + file)
    df_drop = df.drop(columns=['parameter_8', 'parameter_8_value'])
    df_drop.to_csv(save_path + file, index=False)
    print('saved')

for ref_file in ref_files:
    print(ref_file)
    df_ref = pd.read_csv(load_path + ref_file)
    df_ref.to_csv(save_path + ref_file, index=False)
    print('saved')