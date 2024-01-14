import polars as pl
from tqdm import tqdm
import numpy as np
import pickle
from Functions import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")

args = parser.parse_args()

np.random.seed(0)

config = load_config_from_yaml(args.config_path)
test=pl.read_csv(f"{config.input_dir}/test_sequences.csv")#.rename({"sequence_id": "id"})


with open("preds.p",'rb') as f:
    preds=pickle.load(f)


standard_preds=[]

for i in tqdm(range(len(test))):
    length=test['id_max'][i]-test['id_min'][i]+1
    id=test['sequence_id'][i]
    pre_truncated=preds[id][:,[1,0]]
    standard_preds.append(pre_truncated[:length])

standard_preds=np.concatenate(standard_preds)
#arrayed_solution=np.concatenate(arrayed_solution)
sub=pl.read_csv(f"{config.input_dir}/sample_submission.csv")
sub=sub.with_columns(pl.Series(name="reactivity_DMS_MaP", values=standard_preds[:,0]))
sub=sub.with_columns(pl.Series(name="reactivity_2A3_MaP", values=standard_preds[:,1]))

sub=sub.with_columns(
    pl.col("reactivity_DMS_MaP").round(2)
)
sub=sub.with_columns(
    pl.col("reactivity_2A3_MaP").round(2)
)

sub.write_parquet("test.parquet")
