import numpy as np
import csv
from os import path
import polars as pl
import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def drop_pk5090_duplicates(df):
    pk50_filter=df['dataset_name'].str.starts_with('PK50')
    pk90_filter=df['dataset_name'].str.starts_with('PK90')
    no_pk_df=df.filter((~pk50_filter) & (~pk90_filter))
    pk50_df=df.filter(df['dataset_name'].str.starts_with('PK50_AltChemMap_NovaSeq'))
    pk90_df=df.filter(df['dataset_name'].str.starts_with('PK90_Twist_epPCR'))

    assert len(pk50_df)==2729*2
    assert len(pk90_df)==2173*2

    new_df=pl.concat([no_pk_df,pk50_df,pk90_df])

    return new_df

def dataset_dropout(dataset_name,train_indices, dataset2drop):

    #dataset_name=pl.Series(dataset_name)
    dataset_filter=pl.Series(dataset_name).str.starts_with(dataset2drop)
    dataset_filter=dataset_filter.to_numpy()

    dropout_indcies=set(np.where(dataset_filter==False)[0])
    # print(dropout_indcies)
    # exit()


    print(f"number of training examples before droppint out {dataset2drop}")
    print(train_indices.shape)
    before=len(train_indices)

    train_indices=set(train_indices).intersection(set(np.where(dataset_filter==False)[0]))
    train_indices=np.array(list(train_indices))

    print(f"number of training examples after droppint out {dataset2drop}")
    print(len(train_indices))
    after=len(train_indices)
    print(before-after," sequences are dropped")


    # print(set([dataset_name[i] for i in train_indices]))
    # print(len(set([dataset_name[i] for i in train_indices])))
    # exit()

    return train_indices

def get_pl_train(pl_train, seq_length=457):

    print(f"before filtering pl_train has shape {pl_train.shape}")
    pl_train=pl_train.unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])
    print(f"after filtering pl_train has shape {pl_train.shape}")
    #seq_length=206

    label_names=["reactivity_{:04d}".format(number+1) for number in range(seq_length)]
    error_label_names=["reactivity_error_{:04d}".format(number+1) for number in range(seq_length)]

    sequences=pl_train.unique(subset=["sequence_id"],maintain_order=True)['sequence'].to_list()
    sequence_ids=pl_train.unique(subset=["sequence_id"],maintain_order=True)['sequence_id'].to_list()
    labels=pl_train[label_names].to_numpy().astype('float16').reshape(-1,2,seq_length).transpose(0,2,1)
    errors=np.zeros_like(labels).astype('float16')
    SN=pl_train['signal_to_noise'].to_numpy().astype('float16').reshape(-1,2)

    SN[:]=10 # set SN to 10 so they don't get masked

    data_dict = {
        'sequences': sequences,
        'sequence_ids': sequence_ids,
        'labels': labels,
        'errors': errors,
        'SN': SN,
    }

    return data_dict

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

def write_config_to_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def get_distance_mask(L):

    m=np.zeros((L,L))

    for i in range(L):
        for j in range(L):
            if abs(i-j)>0:
                m[i,j]=1/abs(i-j)**2
            elif i==j:
                m[i,j]=1
    return m

class CSVLogger:
    def __init__(self,columns,file):
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()


    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header


    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self,row):
        if len(row)!=len(self.columns):
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file,"a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

if __name__=='__main__':
    print(load_config_from_yaml("configs/sequence_only.yaml"))
