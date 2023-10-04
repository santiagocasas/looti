import pandas as pd
import os

load_path = '../training_data/justClsLensed_10sigma_sig8/'
files = ['lensed_TT.csv.zip', 'lensed_TE.csv.zip', 'lensed_EE.csv.zip']
ref_files = ['lensed_TT_ref.csv.zip', 'lensed_TE_ref.csv.zip', 'lensed_EE_ref.csv.zip']

save_path = '../training_data/justClsLensed_10sigma_nosig8/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in files:
    print(file)
    df = pd.read_csv(load_path + file)
    df_drop = df.drop(columns=['parameter_7', 'parameter_7_value'])
    df_drop.to_csv(save_path + file, index=False)
    print('saved')

for ref_file in ref_files:
    print(ref_file)
    df_ref = pd.read_csv(load_path + ref_file)
    df_ref.to_csv(save_path + ref_file, index=False)
    print('saved')