import polars as pl
from Dataset import *
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from pytorch_ranger import Ranger
import argparse
from accelerate import Accelerator
import time
import json
import matplotlib.pyplot as plt
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig


# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

#from torch.cuda.amp import GradScaler
#from torch import autocast

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
parser.add_argument('--compile', type=str, default="true")

args = parser.parse_args()

np.random.seed(0)

config = load_config_from_yaml(args.config_path)

accelerator = Accelerator(mixed_precision='bf16')

#os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_id)
os.environ["TORCH_DISTRIBUTED_DEBUG"]='DETAIL'


os.system('mkdir logs')
os.system('mkdir models')
os.system('mkdir oofs')
logger=CSVLogger(['epoch','train_loss','val_loss'],f'logs/fold{config.fold}.csv')


with open('data/data_dict.p','rb') as f:
    # data_dict = {
    #     'sequences': sequences,
    #     'sequence_ids': sequence_ids,
    #     'SN': SN,
    # }
    data_dict=pickle.load(f)


with open('data/dataset_name.p','rb') as f:
    dataset_name=pickle.load(f)

data_shape=np.load('data/data_shape.npy')

data_dict['labels']=np.memmap('data/labels.mmap', dtype='float32', mode='r', shape=tuple(data_shape))
data_dict['errors']=np.memmap('data/errors.mmap', dtype='float32', mode='r', shape=tuple(data_shape))



#StratifiedKFold on dataset
kfold=StratifiedKFold(n_splits=config.nfolds,shuffle=True, random_state=0)
fold_indices={}
for i, (train_index, test_index) in enumerate(kfold.split(np.arange(len(dataset_name)),dataset_name)):
    fold_indices[i]=(train_index,test_index)


train_indices=fold_indices[config.fold][0]
val_indices=fold_indices[config.fold][1]

# for data scaling experiments
if config.use_data_percentage<1:
    print(f"Only using {config.use_data_percentage:.02f} of data")
    size=int(config.use_data_percentage*len(train_indices))
    train_indices=np.random.choice(train_indices,size,replace=False)
    print(f"number of sequences in train {len(train_indices)} after subsampling")

if config.use_dirty_data:
    print(f"number of sequences in train {len(train_indices)}")
    train_indices=np.concatenate([train_indices,np.arange(len(dataset_name),len(data_dict['labels']))])
    print(f"number of sequences in train {len(train_indices)} after using dirty data")



if hasattr(config,"dataset2drop"):
    train_indices=dataset_dropout(dataset_name, train_indices, config.dataset2drop)


# if accelerator.is_local_main_process:
#     pl.Config.set_fmt_str_lengths(100)
#     print(data[np.concatenate([train_indices*2,train_indices*2+1])]['dataset_name'].value_counts(sort=True))
#     print(data[np.concatenate([val_indices*2,val_indices*2+1])]['dataset_name'].value_counts(sort=True))
    #print(data[val_indices*2]['dataset_name'].value_counts(sort=True))
#exit()
#train_indices=np.concatenate([train_indices,np.arange(len(train_indices),len(train_indices)+len(dirty_data)//2)])


val_datasets_names=[dataset_name[i] for i in val_indices]

with open("oofs/val_dataset_names.p",'wb+') as f:
    pickle.dump(val_datasets_names,f)

# del data
# del dirty_data

print(f"train shape: {train_indices.shape}")
print(f"val shape: {val_indices.shape}")



#pl_train=pl.read_parquet()
seq_length=data_dict['labels'].shape[1]

# print(seq_length)
# exit()

train_dataset=RNADataset(train_indices,data_dict,k=config.k,
                         flip=config.use_flip_aug)
train_loader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,
                        collate_fn=Custom_Collate_Obj(),num_workers=min(config.batch_size,16),
                        pin_memory=True,
                        prefetch_factor=4,
                        persistent_workers=True)

sample=train_dataset[0]



val_dataset=RNADataset(val_indices,data_dict,train=False,k=config.k)
val_loader=DataLoader(val_dataset,batch_size=config.test_batch_size,shuffle=False,
                        collate_fn=Custom_Collate_Obj(),num_workers=min(config.batch_size,16))


print(accelerator.distributed_type)


model=RibonanzaNet(config)#.cuda()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

optimizer = Ranger(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)
#optimizer = torch.optim.Adam(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)

criterion=torch.nn.L1Loss(reduction='none')
val_criterion=torch.nn.L1Loss(reduction='none')

#.to(accelerator.device)#.cuda().float()

cos_epoch=int(config.epochs*0.75)-1
lr_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(config.epochs-cos_epoch)*len(train_loader)//config.gradient_accumulation_steps)

# for child_name, child in model.named_modules():
#     if 'gate' in child_name:
#         print(child_name)
#     custom_weight_init(child,0)
#exit()
model, optimizer, train_loader, val_loader, lr_schedule= accelerator.prepare(
    model, optimizer, train_loader, val_loader, lr_schedule
)
# Print all the weights and biases
# for name, param in model.named_parameters():
#     # if 'weight' in name:
#     #     print(f"Layer: {name}, Weights: {param.data}")
#     # elif 'bias' in name:
#     #     print(f"Layer: {name}, Biases: {param.data}")
    
#     if "gate" in name:
#         print(f"Layer: {name}, Weights: {param.data}")
#         print(f"Layer: {name}, Biases: {param.data}")
    
if args.compile == 'true':
    model = torch.compile(model,dynamic=False)
#model = model

best_val_loss=np.inf
for epoch in range(config.epochs):

    # training loop
    
    tbar = tqdm(train_loader)
    total_loss=0
    model.train()
    #for batch in tqdm(train_loader):

    for idx, batch in enumerate(tbar):
        
        src=batch['sequence']#.cuda()
        masks=batch['masks'].bool()#.cuda()
        labels=batch['labels']#.cuda()
        SN=batch['SN']

        

        bs=len(labels)
        #batch_attention_mask=batch['attention_mask'].unsqueeze(1)[:,:,:src.shape[-1],:src.shape[-1]]

        loss_masks=batch['loss_masks']#.cuda()
        errors=batch['errors']#.cuda()#.un
#SSH FS test 
        SN=SN.reshape(SN.shape[0],1,SN.shape[1])>=1
        loss_masks=loss_masks*SN

        # print(SN.shape)
        # print(loss_masks.shape)
        # exit()

        #exit()
        #batch_attention_mask=batch['attention_mask']
        #batch_attention_mask=torch.stack([batch_attention_mask[:,:src.shape[-1],:src.shape[-1]],bpp],1)
        SN=batch['SN']
        # print(SN.shape)
        # exit()
        with accelerator.autocast():
            output=model(src,masks)
            loss=criterion(output,labels)#*loss_weight BxLxC
            loss=loss[loss_masks]
            loss=loss.mean()

        accelerator.backward(loss/config.gradient_accumulation_steps)
        
        #loss.backward()
        if (idx + 1) % config.gradient_accumulation_steps == 0:
            #if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            if epoch > cos_epoch:
                lr_schedule.step()

        
        total_loss+=loss.item()
        #exit()
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss/(idx+1)}")
        

        #break
    train_loss=total_loss/(idx+1)
    # if accelerator.is_local_main_process:
    #     if epoch==cos_epoch:
    #     #     torch.save(accelerator.unwrap_model(model).state_dict(),f"models/model{config.fold}_pl_only.pt")
    #     # torch.save(accelerator.unwrap_model(optimizer).state_dict(),f"models/optimizer{config.fold}.pt")
    #         accelerator.save_state("cos_models")

    # validation loop
    model.eval()
    tbar = tqdm(val_loader)
    val_loss=0
    preds=[]
    gts=[]
    print("doing val")
    val_loss_masks=[]

    for idx,batch in enumerate(tbar):
        src=batch['sequence']#.cuda()
        masks=batch['masks'].bool()#.cuda()
        labels=batch['labels']#.cuda()
        bs=len(labels)
        loss_masks=batch['loss_masks']#.cuda()
        src_flipped=src.clone()

        length=batch['length']
        for batch_idx in range(len(src)):
            src_flipped[batch_idx,:length[batch_idx]]=src_flipped[batch_idx,:length[batch_idx]].flip(0)
        src_flipped=src_flipped.clone()

        #with accelerator.autocast():
        with torch.no_grad():
            with accelerator.autocast():
                output=model(src,masks)
                if config.use_flip_aug:
                    flipped_output=model(src_flipped,masks)
                    for batch_idx in range(len(flipped_output)):
                        flipped_output[batch_idx,:length[batch_idx]]=flipped_output[batch_idx,:length[batch_idx]].flip(0)

                    output=(flipped_output+output)/2
        loss=val_criterion(output,labels)[loss_masks]

        L=src.shape[1]
        to_pad=seq_length-L
        #output=output#[loss_masks]
        #labels=labels#[loss_masks]

        output=F.pad(output,(0,0,0,to_pad),value=0)
        labels=F.pad(labels,(0,0,0,to_pad),value=0)
        loss_masks=F.pad(loss_masks,(0,0,0,to_pad),value=0)

        all_output = accelerator.gather(output)
        all_labels = accelerator.gather(labels)
        all_masks = accelerator.gather(loss_masks)

        preds.append(all_output)
        gts.append(all_labels)
        val_loss_masks.append(all_masks)

        loss=loss.mean()
        #loss=torch.sqrt(loss)
        val_loss+=loss.item()

        tbar.set_description(f"Epoch {epoch + 1} Val Loss: {val_loss/(idx+1)}")

        

    #val_loss=val_loss/len(tbar)
        #break
    preds=torch.cat(preds)
    gts=torch.cat(gts)
    val_loss_masks=torch.cat(val_loss_masks)


    # print(accelerator.is_main_process)
    # exit()
    if accelerator.is_main_process:
        val_loss=val_criterion(preds[val_loss_masks],gts[val_loss_masks]).mean().item()

        logger.log([epoch,train_loss,val_loss])

        if val_loss<best_val_loss:
            best_val_loss=val_loss
            #if torch.distributed.get_rank() == 0:
            #torch.save(accelerator.unwrap_model(model).state_dict(),f"models/model{config.fold}.pt")
            #torch.save(model.state_dict(),f"models/model{config.fold}.pt")
            #state_dict=accelerator.get_state_dict(model)
            #torch.save(state_dict,f"models/model{config.fold}.pt")
            # print(accelerator.unwrap_model(model).state_dict())
            # unwrapped_model=accelerator.unwrap_model(model)
            # full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            #     state = accelerator.get_state_dict(unwrapped_model)
            # exit()
            # Using context manager for saving
            # with FSDP.state_dict_type(
            #     model, 
            #     StateDictType.SHARDED_STATE_DICT,
            #     ShardedStateDictConfig(offload_to_cpu=True)  # Optionally offload to CPU
            # ):
            #     state_dict = model.state_dict()
            #     # Each rank saves its own shard
            #     torch.save(state_dict, f"model-shard-{torch.distributed.get_rank()}.pt")

            #accelerator.save_model(model, f"models/model{config.fold}.pt")
            #accelerator.save_state("models")
            data_dict = {
                            "preds": preds.cpu().numpy(),
                            "gts": gts.cpu().numpy(),
                            "val_loss_masks": val_loss_masks.cpu().numpy()
                        }

            # Save to pickle file
            with open(f"oofs/{config.fold}.pkl", "wb+") as file:
                pickle.dump(data_dict, file)
    accelerator.save_state(f"models/epoch_{epoch}")
    # if val_loss<best_val_loss:
    #     accelerator.save_state("ckpt")

    #exit()
    #exit()

if accelerator.is_main_process:
    #torch.save(accelerator.unwrap_model(model).state_dict(),f"models/model{config.fold}_lastepoch.pt")

    end_time = time.time()
    elapsed_time = end_time - start_time

    with open("run_stats.json", 'w') as file:
            json.dump({'Total_execution_time': elapsed_time}, file, indent=4)
