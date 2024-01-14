import polars as pl
from tqdm import tqdm
import numpy as np
import pickle

#df=pl.read_csv("submission_v2.2.0.csv")

#sample_sub=pl.read_csv("/home/exx/Downloads/sample_submission_STANDARD2_NRES207.v2.2.0.csv")
test=pl.read_csv("../../input/test_sequences.csv")#.rename({"sequence_id": "id"})
#test_no_id=pl.read_csv("../../input/v2.2.0/test_sequences.v2.2.0_no_id.csv.gz")

#solution_arrayed=pl.read_csv("../../input/v2.2.0/CONFIDENTIAL/test_data_arrayed_KEEP_CONFIDENTIAL.v2.2.0.csv.gz")
solution=pl.read_csv("../../input/solution_CONFIDENTIAL.v3.3.0.csv",infer_schema_length=0)
#exit()
solution=solution.with_columns(pl.col('reactivity_DMS_MaP').cast(pl.Float64, strict=False))
solution=solution.with_columns(pl.col('reactivity_2A3_MaP').cast(pl.Float64, strict=False))

with open("preds.p",'rb') as f:
    preds=pickle.load(f)

#test_no_id=test_no_id.join(test,on='sequence',how='left')

#df=df.join(test[['id','id_min','id_max']],on='id',how='left').sort('id_min')
 #=["reactivity_{:04d}".format(number+1) for number in range(207)]
# preds=[]
# for idx,grouped_df in tqdm(df.groupby('id',maintain_order=True),total=len(df)//2):
#     length=grouped_df['id_max'][0]-grouped_df['id_min'][0]+1

#preds=df[label_names].to_numpy()
standard_preds=[]
# arrayed_solution=[]
# arrayed_solution_all=solution_arrayed[label_names].to_numpy()
# df_2A3_MaP=df.filter(df['experiment_type']=='2A3_MaP')[label_names].to_numpy()
# df_DMS_MaP=df.filter(df['experiment_type']=='DMS_MaP')[label_names].to_numpy()
for i in tqdm(range(len(test))):
    length=test['id_max'][i]-test['id_min'][i]+1
    id=test['sequence_id'][i]
    pre_truncated=preds[id][:,[1,0]]
    standard_preds.append(pre_truncated[:length])
    #exit()
    #arrayed_solution.append(np.stack([arrayed_solution[i+len(preds)//2],arrayed_solution[i]],-1))
    #exit()
    # preds_tmp=grouped_df[label_names[:length]].to_numpy().transpose()
    # preds.append(preds_tmp)
    #break
standard_preds=np.concatenate(standard_preds)
#arrayed_solution=np.concatenate(arrayed_solution)
assert len(standard_preds)==len(solution)

public_solution=solution.filter(solution['Usage']=='Public')
public_select=public_solution['id'].cast(pl.Int64).to_numpy()
print(f"total amount of public: {len(public_solution)}")
public_gts=public_solution[['reactivity_DMS_MaP','reactivity_2A3_MaP']].to_numpy()#.transpose()
public_score=np.abs(standard_preds[public_select]-public_gts).mean()

private_solution=solution.filter(solution['Usage']=='Private')
private_select=private_solution['id'].cast(pl.Int64).to_numpy()
print(f"total amount of private: {len(private_solution)}")
private_gts=private_solution[['reactivity_DMS_MaP','reactivity_2A3_MaP']].to_numpy()#.transpose()
private_score=np.abs(standard_preds[private_select]-private_gts).mean()

with open("score.txt",'w+') as f:
    f.write(f"Public: {public_score}\n")
    f.write(f"Private: {private_score}\n")

sub=pl.read_csv("../../input/sample_submission.csv")
sub=sub.with_columns(pl.Series(name="reactivity_DMS_MaP", values=standard_preds[:,0]))
sub=sub.with_columns(pl.Series(name="reactivity_2A3_MaP", values=standard_preds[:,1]))

sub=sub.with_columns(
    pl.col("reactivity_DMS_MaP").round(2)
)
sub=sub.with_columns(
    pl.col("reactivity_2A3_MaP").round(2)
)

# Create a Polars Series object from the NumPy array
#series = pl.Series(name="new_column", values=numpy_array)
sub.write_parquet("submission.parquet")
# Set the column in the DataFrame
#df = df.with_column(series)
# exit()
#
# for i in range(207):
#     label=f"reactivity_{i+1:04d}"
#     print(label)
#     df = df.with_column(pl.col(label).round(2))
#
#
# df.write_csv("submission_v2.2.0_rounded.csv")
