from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold
from ranger import Ranger
import argparse
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
import time
import json

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")

args = parser.parse_args()

config = load_config_from_yaml(args.config_path)

accelerator = Accelerator(mixed_precision='fp16')

os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id
os.system('mkdir predictions')
os.system('mkdir plots')
os.system('mkdir subs')

#logger=CSVLogger(['epoch','train_loss','val_loss'],f'logs/fold{config.fold}.csv')

data=pl.read_csv(f"{config.input_dir}/test_sequences.csv")
lengths=data['sequence'].apply(len).to_list()
data = data.with_columns(pl.Series('sequence_length',lengths))
data = data.sort('sequence_length',descending=True)
print(data['sequence_length'])
#sample_sub=pd.read_csv(f"{config.input_dir}/sample_submission_arrayed.v1.0.3.csv")

test_ids=data['sequence_id'].to_list()
sequences=data['sequence'].to_list()
attention_mask=torch.tensor(get_distance_mask(max(lengths))).float()
#sequence_ids=data.unique(subset=["sequence_id"],maintain_order=True)['sequence_id'].to_list()
data_dict={'sequences':sequences,
           'sequence_ids': test_ids,
           "attention_mask":attention_mask}
assert len(test_ids)==len(data)
#exit()
#exit()

val_dataset=TestRNAdataset(np.arange(len(data)),data_dict,k=config.k)
val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,
                        collate_fn=Custom_Collate_Obj_test(),num_workers=min(config.batch_size,32))

# val_dataset[0]
# exit()
print(accelerator.distributed_type)
models=[]

for i in range(1):
    model=RibonanzaNet(config)#.cuda()
    model.eval()
    model.load_state_dict(torch.load(f"models/model{i}.pt",map_location='cpu'))
    models.append(model)

#exit()

model, val_loader= accelerator.prepare(
    model, val_loader
)

#print(val_dataset.max_len)
# print(attention_mask.device)
# exit()
tbar = tqdm(val_loader)
val_loss=0
preds=[]
model.eval()
for idx, batch in enumerate(tbar):
    src=batch['sequence']#.cuda()
    masks=batch['masks']#.bool().cuda()
    bs=len(src)

    src_flipped=src.clone()



    length=batch['length']
    #batch_attention_mask_flipped=batch_attention_mask.clone()
    for batch_idx in range(len(src)):
        src_flipped[batch_idx,:length[batch_idx]]=src_flipped[batch_idx,:length[batch_idx]].flip(0)
        #batch_attention_mask_flipped[batch_idx,:,:length[batch_idx],:length[batch_idx]]=batch_attention_mask_flipped[batch_idx,:,:length[batch_idx],0:length[batch_idx]].flip(-1).flip(-2)
        #bpp_flipped[batch_idx,:length[batch_idx],:length[batch_idx]]=bpp_flipped[batch_idx,:length[batch_idx],:length[batch_idx]].flip(-1).flip(-2)
    #batch_attention_mask_flipped=torch.stack([attention_mask.expand(bs,*attention_mask.shape)[:,:src.shape[-1],:src.shape[-1]],bpp_flipped],1)

    with torch.no_grad():
        with accelerator.autocast():
            output=[]
            for model in models:
                output.append(model(src,masks))
                if config.use_flip_aug:
                    flipped_output=model(src_flipped,masks)
                    for batch_idx in range(len(flipped_output)):
                        flipped_output[batch_idx,:length[batch_idx]]=flipped_output[batch_idx,:length[batch_idx]].flip(0)

                    output.append(flipped_output)
            output=torch.stack(output).mean(0)
    #exit()
    output = accelerator.pad_across_processes(output,1)
    all_output = accelerator.gather(output).cpu().numpy()
    preds.append(all_output)

if accelerator.is_local_main_process:
    import pickle

    preds_dict={}

    for i,id in tqdm(enumerate(test_ids)):
        batch_number=i//(config.test_batch_size*accelerator.num_processes)
        in_batch_index=i%(config.test_batch_size*accelerator.num_processes)
        preds_dict[id]=preds[batch_number][in_batch_index]

    with open("preds.p",'wb+') as f:
        pickle.dump(preds_dict,f)

    end_time = time.time()
    elapsed_time = end_time - start_time

    with open("inference_stats.json", 'w') as file:
            json.dump({'Total_execution_time': elapsed_time}, file, indent=4)
