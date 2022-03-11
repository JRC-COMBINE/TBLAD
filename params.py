'''
This module has a list of all parameters used in other modules
'''

### Classifier parameters 
# number of forests

N_forests = 5

# number of estimators per forest
N_estimators = 20


### preprocessor params 
data_path = ['../../data/raw/2020_04_15_ions_datasets/Cl-/','../../data/raw/2020_04_15_ions_datasets/Br-/','../../data/raw/2020_04_15_ions_datasets/NO3-/','../../data/raw/2020_04_15_ions_datasets/ClO4-/','../../data/raw/2020_04_15_ions_datasets/SCN-/']

chunk_size = 10000

dataframe_path = './dataframe/'
feature_path = './Features/'
output_path = './images/'

## path files of raw amperometric data
path_raw_Br = '../../data/raw/2020_04_15_ions_datasets/Br-/'
path_raw_Cl = '../../data/raw/2020_04_15_ions_datasets/Cl-/'
path_raw_NO3 = '../../data/raw/2020_04_15_ions_datasets/NO3-/'
path_raw_ClO4 = '../../data/raw/2020_04_15_ions_datasets/ClO4-/'
path_raw_SCN = '../../data/raw/2020_04_15_ions_datasets/SCN-/'

