from accelerate.utils import merge_fsdp_weights

# Our weights are saved usually in a `pytorch_model_fsdp_{model_number}` folder
merge_fsdp_weights("ckpt/pytorch_model_fsdp_0", "ckpt", safe_serialization=False)
