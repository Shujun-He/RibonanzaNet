import polars as pl
from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
from accelerate import Accelerator
import time
import json
import matplotlib.pyplot as plt
import numpy as np
#from torch.cuda.amp import GradScaler
#from torch import autocast

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")

args = parser.parse_args()

np.random.seed(0)

config = load_config_from_yaml(args.config_path)


os.system('mkdir data')

#exit()

#data=pd.read_csv(f"{config.input_dir}/train_data.v2.3.0.csv.gz")
data=pl.read_csv(f"{config.input_dir}/train_data.csv")

#new_ids=
#data=data.with_columns(pl.Series(name="id", values=[id+"_"+exp_type for id, exp_type in zip(data['sequence_id'],data['experiment_type'])]))

pl.Config.set_fmt_str_lengths(100)
# print(data['dataset_name'].value_counts(sort=True))
# print(data['dataset_name'].value_counts(sort=True))
# exit()

data=drop_pk5090_duplicates(data)

print("before dropping duplicates data shape is:",data.shape)
data=data.unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])
print("after dropping duplicates data shape is:",data.shape)
#data=data.sort(["signal_to_noise"],descending=True).unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])

n_sequences_total=len(data)//2
#get necessary data as lists and numpy arrays
seq_length=206

#filter out a sequence if min SN is smaller than 1
SN=data['signal_to_noise'].to_numpy().astype('float32').reshape(-1,2)
SN=SN.min(-1)
SN=np.repeat(SN,2)
print("before filtering data shape is:",data.shape)
dirty_data=data.filter((SN<=1))
data=data.filter(SN>1)
print("after filtering data shape is:",data.shape)
print("direty data shape is:",dirty_data.shape)

# get sequences where one of 2A3/DMS has SN>1
dirty_SN=dirty_data['signal_to_noise'].to_numpy().astype('float32').reshape(-1,2)
dirty_SN=dirty_SN.max(-1)
dirty_SN=np.repeat(dirty_SN,2)
dirty_data=dirty_data.filter(dirty_SN>1)
print("after filtering dirty_data shape is:",dirty_data.shape)


label_names=["reactivity_{:04d}".format(number+1) for number in range(seq_length)]
error_label_names=["reactivity_error_{:04d}".format(number+1) for number in range(seq_length)]

sequences=data.unique(subset=["sequence_id"],maintain_order=True)['sequence'].to_list()
sequence_ids=data.unique(subset=["sequence_id"],maintain_order=True)['sequence_id'].to_list()
labels=data[label_names].to_numpy().astype('float32').reshape(-1,2,206).transpose(0,2,1)
errors=data[error_label_names].to_numpy().astype('float32').reshape(-1,2,206).transpose(0,2,1)
SN=data['signal_to_noise'].to_numpy().astype('float32').reshape(-1,2)
dataset_name=data['dataset_name'].to_list()
dataset_name=[dataset_name[i*2].replace('2A3','NULL').replace('DMS','NULL') for i in range(len(data)//2)]


data_dict = {
    'sequences': sequences,
    'sequence_ids': sequence_ids,
    'labels': labels,
    'errors': errors,
    'SN': SN,
}

if config.use_dirty_data:
    print("using sequences where one of 2A3/DMS has SN>1")
    data_dict['sequences']+=dirty_data.unique(subset=["sequence_id"],maintain_order=True)['sequence'].to_list()
    data_dict['sequence_ids']+=dirty_data.unique(subset=["sequence_id"],maintain_order=True)['sequence_id'].to_list()
    data_dict['labels']=np.concatenate([data_dict['labels'],
                            dirty_data[label_names].to_numpy().astype('float32').reshape(-1,2,206).transpose(0,2,1)])
    data_dict['errors']=np.concatenate([data_dict['errors'],
                            dirty_data[error_label_names].to_numpy().astype('float32').reshape(-1,2,206).transpose(0,2,1)])
    data_dict['SN']=np.concatenate([data_dict['SN'],
                            dirty_data['signal_to_noise'].to_numpy().astype('float32').reshape(-1,2)])

    # print(f"number of sequences in train {len(train_indices)}")
    # train_indices=np.concatenate([train_indices,np.arange(len(data)//2,len(data)//2+len(dirty_data)//2)])
    # print(f"number of sequences in train {len(train_indices)} after using dirty data")



#save small objects in a pickle file
with open('data/data_dict.p','wb+') as f:
    save_data_dict = {
        'sequences': data_dict['sequences'],
        'sequence_ids': data_dict['sequence_ids'],
        'SN': data_dict['SN'],
    }
    pickle.dump(save_data_dict,f)

with open('data/dataset_name.p','wb+') as f:
    pickle.dump(dataset_name,f)



#save labels 
filename = 'data/labels.mmap'  # Specify the path where the memmap file should be stored
dtype = 'float32'       # Change this to match the dtype of your labels array
mode = 'w+'             # Write mode, will create or overwrite existing file
shape = data_dict['labels'].shape  # Shape of the array

# Create the memmap array
mmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

mmap_array[:] = data_dict['labels']
mmap_array.flush()

#save errors 
filename = 'data/errors.mmap'  # Specify the path where the memmap file should be stored
dtype = 'float32'       # Change this to match the dtype of your labels array
mode = 'w+'             # Write mode, will create or overwrite existing file
#shape = (167671, 206, 2)  # Shape of the array

# Create the memmap array
mmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

mmap_array[:] =  data_dict['errors']
mmap_array.flush()

np.save('data/data_shape',shape)

